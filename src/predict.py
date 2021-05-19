import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from preprocess import predict_preprocess
from model import BPR_model, BCE_model


num_user, num_item, train_data, mask = predict_preprocess()

device = torch.device('cpu')



# Load Model
model_1 = BPR_model(num_user, num_item, 512)
model_2 = BPR_model(num_user, num_item, 256)
model_3 = BPR_model(num_user, num_item, 1024)
# model_4 = BPR_model(num_user, num_item, 1024)
# model_5 = BPR_model(num_user, num_item, 768)

# model_1 = BCE_model(num_user, num_item, 2048)
# model_2 = BCE_model(num_user, num_item, 1024)
# model_3 = BCE_model(num_user, num_item, 1024)
# model_4 = BCE_model(num_user, num_item, 512)
# model_5 = BCE_model(num_user, num_item, 2048)

model_dir = "./model/"

model_1.load_state_dict(torch.load(model_dir + 'BPR-model_512.pkct', map_location=device))
model_2.load_state_dict(torch.load(model_dir + 'BPR-model_256.pkct', map_location=device))
model_3.load_state_dict(torch.load(model_dir + 'BPR-model_1024.pkct', map_location=device))
# model_4.load_state_dict(torch.load(model_dir + 'BPR-model_4.pkct', map_location=device))
# model_5.load_state_dict(torch.load(model_dir + 'model_5.pkct', map_location=device))

# model_1.load_state_dict(torch.load(model_dir + 'BCE-model_1.pkct', map_location=device))
# model_2.load_state_dict(torch.load(model_dir + 'BCE-model_2.pkct', map_location=device))
# model_3.load_state_dict(torch.load(model_dir + 'BCE-model_3.pkct', map_location=device))
# model_2.load_state_dict(torch.load(model_dir + 'BCE-model_4.pkct', map_location=device))
# model_1.load_state_dict(torch.load(model_dir + 'BCE-model_5.pkct', map_location=device))

# Predicting & ensemble
pred = []
model_1.eval()
# model_2.eval()
# model_3.eval()
# model_4.eval()
# model_5.eval()

total_item = []
for i in range(num_item):
    total_item.append(i)
    
for i in range(num_user):
    ap = 0
    correct = 0
    count = 0
    item = torch.LongTensor(total_item)
    with torch.no_grad():
        output_1 = model_1(torch.LongTensor([i]), item.unsqueeze(0), test=True)
        # output_2 = model_2(torch.LongTensor([i]), item.unsqueeze(0), test=True)
        # output_3 = model_3(torch.LongTensor([i]), item.unsqueeze(0), test=True)
        # output_4 = model_4(torch.LongTensor([i]), item.unsqueeze(0), test=True)
        # output_5 = model_5(torch.LongTensor([i]), item.unsqueeze(0), test=True)
    # output = output_3 + output_4
    output =  output_1 #+ output_2 + output_3 + output_4 + output_5
    ranking, idx = torch.topk(output, 500)
    idx = idx.squeeze(1).squeeze(0).tolist()
    
    count = 0
    pred_idx = []
    for j in range(500):
        if count == 50:
            break
        if (i, idx[j]) not in mask:
            count += 1
            pred_idx.append(idx[j])
    pred.append(pred_idx)

# Write pred to output csv
with open(sys.argv[1], 'w') as f:
    f.write('UserId,ItemId\n')
    for i, user_pred in enumerate(pred):
        f.write('{},'.format(i))
        first = 0
        for item in user_pred:
            if first == 0:
                first = 1
            else:
                f.write(' ')
            f.write('{}'.format(item))
        f.write('\n')