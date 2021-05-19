import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data, num_item, neg_num, mask=None):
        super(Dataset, self).__init__()
        self.raw_data = data
        self.num_item = num_item
        self.neg_num = neg_num
        self.mask = mask
        self.init = True
        self.sample()

    def __len__(self):
        return len(self.data)

    def sample(self):
        self.data = []
        for x in self.raw_data:
            user, items = x[0], x[1]
            for i in items:
                for _ in range(self.neg_num):
                    j = np.random.randint(self.num_item)
                    while (user, j) in self.mask:
                        j = np.random.randint(self.num_item)
                    self.data.append([user, i, j])
                
    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]
        item_j = self.data[idx][2]
        return user, item_i, item_j 
