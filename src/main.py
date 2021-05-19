import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataset import Dataset
from preprocess import preprocess
from model import BPR_model, BCE_model
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=3072, help="batch size for training") 
parser.add_argument("--num_epochs", type=int, default=50, help="training epoches") 
parser.add_argument("--embed_dim", type=int, default=1024, help="predictive factors numbers in the model") 
parser.add_argument("--neg_num", type=int, default=30, help="# of negative samples for one positive sample") 
args = parser.parse_args()

seed = 65537
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

num_user, num_item, train_data, test_data, mask = preprocess()
train_dataset = Dataset(train_data, num_item, args.neg_num, mask)
valid_dataset = Dataset(test_data, num_item, args.neg_num, mask)

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=1024, shuffle=False)
# train_loader.dataset.sample()
# valid_loader.dataset.sample()

print("before model")
model = BPR_model(num_user, num_item, args.embed_dim).cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=4)

# warmup_steps = 2000
num_epoch = args.num_epochs
best_valid_loss = 5000
total_steps = len(train_loader) *  num_epoch

# warmup scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

for epoch in range(num_epoch):
    
    if epoch % 10 == 0: # change opt 
        train_loader.dataset.sample()
        valid_loader.dataset.sample()
    start_time = time.time()
    total_loss = 0.0
    model.train() 
    # for param_group in optimizer.param_groups:
    #     print("lr: {}".format(param_group['lr']))

    # ========= training ============    
    for u, i, j in train_loader:
        u = torch.LongTensor(u).cuda()
        i = torch.LongTensor(i).cuda()
        j = torch.LongTensor(j).cuda()
        loss = model(u, i, j)
        loss.backward()

        total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)), end=" ")
    print("avg loss:", total_loss / len(train_loader))

    # ========= validation ============
    model.eval()
    if epoch + 1 == num_epoch:
        pred = []
    
    total_loss = 0.0
    for u, i, j in (valid_loader):
        u = torch.LongTensor(u).cuda()
        i = torch.LongTensor(i).cuda()
        j = torch.LongTensor(j).cuda()
        with torch.no_grad():
            loss = model(u, i, j)
        
        total_loss += loss.item()
    print("Valid Loss:{}".format(total_loss/len(valid_loader)))
    scheduler.step(total_loss/len(valid_loader))

    if total_loss/len(valid_loader) < best_valid_loss:
        # best_valid_loss = total_loss/len(valid_loader)
        print("saved!")
        torch.save(model.state_dict(), './model/BPR-model_1024.pkct')
