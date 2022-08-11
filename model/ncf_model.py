# Define Neural Collaborative Filtering model
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, cust_num, prod_num, feature_num, num_layers, dropout_rate):
        super(NCF, self).__init__()
        self.embed_cust_GMF = nn.Embedding(cust_num, feature_num)
        self.embed_prod_GMF = nn.Embedding(prod_num, feature_num)
        self.embed_cust_MLP = nn.Embedding(cust_num, feature_num * (2 ** (num_layers - 1)))
        self.embed_prod_MLP = nn.Embedding(prod_num, feature_num * (2 ** (num_layers - 1)))
        self.dropout = nn.Dropout(dropout_rate)

        self.MLP_layer = nn.Sequential()
        for i in range(num_layers):
            input_size = feature_num * (2 ** (num_layers - i))
            self.MLP_layer.add_module("dropout%d" % (i), nn.Dropout(dropout_rate))
            self.MLP_layer.add_module("linear%d" % (i), nn.Linear(input_size, input_size // 2))
            self.MLP_layer.add_module("relu%d" % (i), nn.ReLU())
        self.NeuMF_layer = nn.Linear(feature_num * 2, 1)

    def forward(self, cust, prod):
        embed_cust_gmf = self.embed_cust_GMF(cust)
        embed_prod_gmf = self.embed_prod_GMF(prod)
        output_gmf = embed_cust_gmf * embed_prod_gmf

        embed_cust_mlp = self.embed_cust_MLP(cust)
        embed_prod_mlp = self.embed_prod_MLP(prod)
        output_mlp = self.MLP_layer(torch.cat((embed_cust_mlp, embed_prod_mlp), -1))

        final = self.NeuMF_layer(torch.cat((output_gmf, output_mlp), -1))
        return final.view(-1)