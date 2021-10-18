# rdrop单独
# import
import os
import math
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer
from transformers import BertConfig, BertModel

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

log_dir = '/home/dango/multimodal/contrastive/huiyi/bertrdrop_log/'
ren_train_file = '/home/dango/multimodal/contrastive/huiyi/data/ren_train.txt'
ren_test_file = '/home/dango/multimodal/contrastive/huiyi/data/ren_test.txt'
EPOCHS = 999
BATCH = 32
MAX_LEN = 100
CLIP = 1.0
LR = 5e-4

# data
def data_set(catagory='test', save=False):
    if catagory == 'test':
        ren_file = ren_test_file
    else:
        ren_file = ren_train_file
    data = []
    with open(ren_file, 'r') as rf:
        lines = rf.readlines()[1:]
        for line in lines:
            label = []
            text = line.strip().split('|')[0]
            for i in range(9):
                label.append(int(line.strip().split('|')[i+1]))
            data.append([text, label])
    return data

train_set = data_set('train')
test_set = data_set('test')

def data_loader(data_set, batch_size):
    random.shuffle(data_set)
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            text = data_set[count][0]
            label = data_set[count][1]
            batch.append((text, label))
            batch.append((text, label))
            count += 1
        yield batch

# model
class SimCSE(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext"):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained)
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state[:, 0]
#         return torch.mean(output.last_hidden_state, 1)
#         return torch.max(output.last_hidden_state, 1)[0]

class final_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.simcse = SimCSE()
        self.classifier = nn.Linear(768, 9)  # (no need sigmoid)
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(self.simcse(input_ids, attention_mask, token_type_ids))  # (batch, 9)

# run
def multi_circle_loss(y_pred, y_true):
    y_pred, y_true = y_pred[::2], y_true[::2]
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1], dtype = torch.float)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss

def kl_div_loss(y_pred):
    odd = torch.nn.functional.kl_div(y_pred[::2], y_pred[1::2], reduction='batchmean', log_target=True)
    even = torch.nn.functional.kl_div(y_pred[1::2], y_pred[::2], reduction='batchmean', log_target=True)
    return torch.abs(odd) + torch.abs(even)

def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        text, label = zip(*batch)
        token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt")
        token, label = token.to(device), torch.cuda.LongTensor(label)
        logits_clsf = model(**token)
        multi_loss = multi_circle_loss(logits_clsf, label)
        kl_loss = kl_div_loss(logits_clsf)
        loss = multi_loss.mean() + kl_loss/4
#         loss = multi_loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            text, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt")
            token, label = token.to(device), torch.cuda.LongTensor(label)
            logits_clsf = model(**token)
            multi_loss = multi_circle_loss(logits_clsf, label)
            kl_loss = kl_div_loss(logits_clsf)
            loss = multi_loss.mean() + kl_loss/4
#             loss = multi_loss.mean()
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss, count, epoch_loss / count

def run(model, train_list, valid_list, batch_size, learning_rate, epochs, name):
    log_file = log_dir+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
#     my_list = ['simcse.encoder.pooler.dense.weight', 'simcse.encoder.pooler.dense.bias', 'classifier.weight', 'classifier.bias']
    my_list = ['classifier.weight', 'classifier.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
#     optimizer = optim.AdamW([{'params': base_params}, {'params': params, 'lr': learning_rate * 10}], lr=learning_rate)
    optimizer = optim.AdamW([{'params': base_params, 'lr': 1e-5}, {'params': params, 'lr': 1e-2}])
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, batch_size)
        valid_iterator = data_loader(valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        _, _, valid_loss = valid(model, valid_iterator)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch},{train_loss: 2.2f},{valid_loss: 2.2f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list) and valid_loss > 0.009:
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 3:
                break
    writer.close()

# train
random.shuffle(train_set)
model_1 = final_model().to(device)
valid_list = train_set[:6720]
train_list = train_set[6720:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_1')

model_2 = final_model().to(device)
valid_list = train_set[6720:13440]
train_list = train_set[:6720] + train_set[13440:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_2')

model_3 = final_model().to(device)
valid_list = train_set[13440:20160]
train_list = train_set[:13440] + train_set[20160:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_3')

model_4 = final_model().to(device)
valid_list = train_set[20160:26880]
train_list = train_set[:20160] + train_set[26880:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='model_4')
