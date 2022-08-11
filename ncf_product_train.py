import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.data_preprocessing import *
from model.ncf_model import *
from utils.functions import *

def ncf_product_train():
    # CUDA, GPU setting
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get configs and create NCF model
    cfg = get_configs()
    ncf_model = NCF(cfg.CUST_NUM, cfg.PROD_NUM, cfg.FEATURE_NUM, cfg.NUM_LAYERS, cfg.DROPOUT_RATE)
    ncf_model.to(device)
    ncf_model.train()

    # Declare loss function, optimizer, dataset/dataloader, tensorboard
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(ncf_model.parameters(), lr=cfg.LR)
    user_item_matrix = create_cust_prod_matrix(cfg.CREATE_MATRIX, cfg.PATH)
    traindataset = MyDataSet(user_item_matrix)
    traindataloader = DataLoader(traindataset[:int(user_item_matrix.shape[0] * 0.9)], batch_size=256, shuffle=True)
    writer = SummaryWriter(cfg.PATH + 'tensorboard/ncf_product/exp' + str(cfg.EXPNUM))

    # Train the model
    print("############### Begin training for NCF_product ###############")
    for epoch in range(cfg.EPOCH):
        i = 0
        for (cust, prod), label in traindataloader:
            optimizer.zero_grad()
            my_output = ncf_model(cust, prod)
            loss = criterion(my_output, label.float().view(-1))
            writer.add_scalar('Training loss', loss, epoch * len(traindataloader) + i)
            if i % 1000 == 0:
                print("Epoch: %d, i: %d, loss: %f" % (epoch, i, loss.cpu().detach().numpy()))
            loss.backward()
            optimizer.step()
            i += 1
        torch.save(ncf_model.state_dict(), cfg.PATH + 'save/ncf_product.pth')

if __name__ == "__main__":
    ncf_product_train()