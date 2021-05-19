import torch.nn as nn
import torch
import torch.nn.functional as F

class BPR_model(nn.Module):
    def __init__(self, num_user, num_item, embed_dim):
        super(BPR_model, self).__init__()
        self.user_embed= nn.Embedding(num_user, embed_dim)
        self.item_embed = nn.Embedding(num_item, embed_dim)
        nn.init.orthogonal_(self.user_embed.weight)
        nn.init.orthogonal_(self.item_embed.weight)

    def forward(self, u, i, j=None, test=False):
        u = self.user_embed(u)
        i = self.item_embed(i)
        if test == False:
            prediction_i = torch.bmm(u.unsqueeze(1), i.unsqueeze(2)).squeeze(2).squeeze(1)
        else:
            prediction_i = torch.bmm(u.unsqueeze(1), i.permute(0, 2, 1))
            return torch.sigmoid(prediction_i)

        j = self.item_embed(j)

        prediction_j = torch.bmm(u.unsqueeze(1), j.unsqueeze(2)).squeeze(2).squeeze(1)
        log_prob = F.logsigmoid(prediction_i - prediction_j).sum()

        return -log_prob

class BCE_model(nn.Module):
    def __init__(self, num_user, num_item, embed_dim):
        super(BCE_model, self).__init__()
        self.user_embed= nn.Embedding(num_user, embed_dim)
        self.item_embed = nn.Embedding(num_item, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, u, i, j=None, test=False):
        u = self.user_embed(u)
        i = self.item_embed(i)
        if test == False:
            prediction_i = torch.bmm(u.unsqueeze(1), i.unsqueeze(2)).squeeze(2).squeeze(1)
        else:
            prediction_i = torch.bmm(u.unsqueeze(1), i.permute(0, 2, 1))
            return torch.sigmoid(prediction_i)


        j = self.item_embed(j)
        prediction_j = torch.bmm(u.unsqueeze(1), j.unsqueeze(2)).squeeze(2).squeeze(1)
        BCE_i = torch.log(torch.sigmoid(prediction_i)).sum()
        BCE_j = torch.log(1- torch.sigmoid(prediction_j)).sum()
        # BCE_i = prediction_i.sigmoid().log().sum()
        # BCE_j = torch.add(torch.multiply(prediction_j.sigmoid(), -1), 1).log().sum()
        # print(BCE_i)
        # print(BCE_j)
        log_prob = BCE_i + BCE_j

        return -log_prob