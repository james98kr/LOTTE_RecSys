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

def ncf_service_test():
    # CUDA, GPU setting
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get configs and create NCF model
    cfg = get_configs()
    ncf_model = NCF(cfg.CUST_NUM, cfg.SERV_NUM, cfg.FEATURE_NUM, cfg.NUM_LAYERS, cfg.DROPOUT_RATE)
    ncf_model.load_state_dict(torch.load(cfg.SAVED_MODEL_TO_TEST))
    ncf_model.to(device)
    ncf_model.eval()

    # Test on pre-selected test dataloader (10% of total dataset)
    print("############### Begin testing for NCF_service ###############")
    user_item_matrix = create_cust_serv_matrix(cfg.CREATE_MATRIX, cfg.PATH)
    testdataset = MyDataSet(user_item_matrix[int(user_item_matrix.shape[0] * 0.9):])
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=True)
    total_error = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    i = 0
    for (cust, serv), label in testdataloader:
        my_output = ncf_model(cust, serv)
        loss = float((my_output.unsqueeze(-1) - label.float()) ** 2)
        total_error += loss
        if float(my_output) >= 0 and float(label) == 1:
            tp += 1
        elif float(my_output) < 0 and float(label) == 1:
            fn += 1
        elif float(my_output) < 0 and float(label) == 0:
            tn += 1
        elif float(my_output) >= 0 and float(label) == 0:
            fp += 1
        else:
            raise Exception("A huge error!")
        if i % 1000 == 0 and i != 0:
            print("i: %d, total error: %f, false negative: %d, false positive: %d, modified error rate: %f" % (i, total_error / i, fn, fp, (fp + fn) / i))
        i += 1
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("Total MSE error is %f,\nModified error rate is %f,\nTP: %d, TN: %d, FP: %d, FN: %d,\nPrecision: %f,\nRecall: %f,\nF1 Score: %f" % \
        (total_error / len(testdataloader), (fp + fn) / len(testdataloader), tp, tn, fp, fn, precision, recall, f1))

if __name__ == "__main__":
    ncf_service_test()