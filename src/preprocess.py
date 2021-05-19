import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(valid_ratio = 0.01):
    path_to_input = "./data/train.csv"

    f = open(path_to_input)
    f.readline()
    user_num = -1
    item_num = -1
    raw_data = []
    for i, line in enumerate(f):
        user, items = int(line.split(',')[0]), list(map(int, line.split(',')[1].split(' ')))
        user_num = user
        item_num = max(item_num, max(items))
        raw_data.append((user, items))
    # print(train_data[0][1])
    user_num += 1
    item_num += 1
    print("user num: ", user_num, "item num: ", item_num)

    train_data = []
    test_data = []
    for user, items in raw_data:
        train_data.append((user, items[1:]))
        test_data.append((user, items[:1]))

    mask = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for user, items in raw_data:
        for item in items:
            mask[user, item] = 1.0
    return user_num, item_num, train_data, test_data, mask

def predict_preprocess():
    path_to_input = "./data/train.csv"

    f = open(path_to_input)
    f.readline()
    user_num = -1
    item_num = -1
    raw_data = []
    for i, line in enumerate(f):
        user, items = int(line.split(',')[0]), list(map(int, line.split(',')[1].split(' ')))
        user_num = user
        item_num = max(item_num, max(items))
        raw_data.append((user, items))
    # print(train_data[0][1])
    user_num += 1
    item_num += 1
    print("user num: ", user_num, "item num: ", item_num)


    mask = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for user, items in raw_data:
        for item in items:
            mask[user, item] = 1.0
    return user_num, item_num, raw_data, mask
