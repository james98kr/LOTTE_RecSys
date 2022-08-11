from datetime import datetime
from tabulate import tabulate
from colorama import Fore
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_preprocessing import *
from model.ncf_model import *
from model.pd_model import *
from utils.functions import *

tabulate.WIDE_CHARS_MODE = False
CURMONTH = datetime.today().month
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class RecSys:
    def __init__(self, cfg):
        df1 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_01_DEMO.csv')
        df2 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_02_PDDE.csv', low_memory=False)
        df3 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_03_COP_U.csv')
        df4 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_04_PD_CLAC.csv')
        df5 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_05_BR.csv')
        df6 = pd.read_csv(cfg.PATH + 'data/raw/LPOINT_BIG_COMP_06_LPAY.csv')
        self.dfmerge124 = DF_MERGE_124(df1, df2, df4)
        self.dfmerge135 = DF_MERGE_135(df1, df3, df5)
        self.ncf_product_model = NCF(cfg.CUST_NUM, cfg.PROD_NUM, cfg.FEATURE_NUM, cfg.NUM_LAYERS, cfg.DROPOUT_RATE).to(DEVICE)
        self.ncf_service_model = NCF(cfg.CUST_NUM, cfg.SERV_NUM, cfg.FEATURE_NUM, cfg.NUM_LAYERS, cfg.DROPOUT_RATE).to(DEVICE)
        self.ncf_product_model.load_state_dict(torch.load(cfg.NCF_PRODUCT_MODEL))
        self.ncf_service_model.load_state_dict(torch.load(cfg.NCF_SERVICE_MODEL))
        
    def get_ncf_product_rec(self, cust_index, top=50):
        if isinstance(cust_index, str):
            cust_index = self.dfmerge124.cust_dict2.get(cust_index)
        if cust_index == None or cust_index not in self.dfmerge124.inverse_cust_dict2:
            return None
        cust_index = torch.Tensor([cust_index]).long().to(DEVICE)
        ret = []
        for prod_index in self.dfmerge124.inverse_prod_dict:
            output = torch.sigmoid(self.ncf_product_model(cust_index, torch.Tensor([prod_index]).long().to(DEVICE)))
            ret.append([output, prod_index])
        ret = sorted(ret, key=lambda x:x[0], reverse=True)
        ret = [self.dfmerge124.inverse_prod_dict[ret[i][1]] for i in range(top)]
        return ret

    def get_ncf_service_rec(self, cust_index, top=50):
        if isinstance(cust_index, str):
            cust_index = self.dfmerge135.cust_dict3.get(cust_index)
        if cust_index == None or cust_index not in self.dfmerge135.inverse_cust_dict3:
            return None
        cust_index = torch.Tensor([cust_index]).long().to(DEVICE)
        ret = []
        for serv_index in self.dfmerge135.inverse_serv_dict:
            output = torch.sigmoid(self.ncf_service_model(cust_index, torch.Tensor([serv_index]).long().to(DEVICE)))
            ret.append([output, serv_index])
        ret = sorted(ret, key=lambda x:x[0], reverse=True)
        ret = [self.dfmerge135.inverse_serv_dict[ret[i][1]] for i in range(top)]
        return ret

    def get_pd_product_rec(self, cust_index, month=CURMONTH):
        if isinstance(cust_index, str):
            cust_index = self.dfmerge124.cust_dict2.get(cust_index)
        if cust_index == None or cust_index not in self.dfmerge124.inverse_cust_dict2:
            return None
        out1 = list(self.dfmerge124.find_relevant_products(cust_index, month)['pd_c'])
        out2 = list(self.dfmerge124.find_top_products(cust_index)['pd_c'])
        return out1 + out2

    def get_pd_service_rec(self, cust_index, month=CURMONTH):
        if isinstance(cust_index, str):
            cust_index = self.dfmerge135.cust_dict3.get(cust_index)
        if cust_index == None or cust_index not in self.dfmerge135.inverse_cust_dict3:
            return None
        out1 = list(self.dfmerge135.find_relevant_servs(cust_index, month)['br_c'])
        out2 = list(self.dfmerge135.find_top_servs(cust_index)['br_c'])
        return out1 + out2
    
    def get_products(self, cust_index, month=CURMONTH, num_rows=50):
        ncf_ret = self.get_ncf_product_rec(cust_index)
        pd_ret = self.get_pd_product_rec(cust_index, month)
        if ncf_ret == None or pd_ret == None:
            print("There is no product purchase history information for customer %s, so unable to make recommendations!" % (cust_index))
            return

        ncf_ret_df = self.dfmerge124.df4[self.dfmerge124.df4['pd_c'].isin(ncf_ret)]
        sorterIndex = dict(zip(ncf_ret, range(len(ncf_ret))))
        ncf_ret_df['rank'] = ncf_ret_df['pd_c'].map(sorterIndex)
        ncf_ret_df = ncf_ret_df.sort_values(by='rank')
        ncf_ret_df = ncf_ret_df[['pd_c', 'pd_nm']].reset_index(drop=True).head(num_rows)
        ncf_ret_df = ncf_ret_df.rename(columns={'pd_nm': 'pd_nm                   '})

        pd_ret_df = self.dfmerge124.df4[self.dfmerge124.df4['pd_c'].isin(pd_ret)]
        sorterIndex = dict(zip(pd_ret, range(len(pd_ret))))
        pd_ret_df['rank'] = pd_ret_df['pd_c'].map(sorterIndex)
        pd_ret_df = pd_ret_df.sort_values(by='rank')
        pd_ret_df = pd_ret_df[['pd_c', 'pd_nm']].reset_index(drop=True).head(num_rows)
        pd_ret_df = pd_ret_df.rename(columns={'pd_nm': 'pd_nm                   '})

        print(Fore.BLUE + "\nTop Recommendable Products:" + Fore.WHITE)
        print(tabulate(ncf_ret_df, headers = 'keys', tablefmt = 'rst'))
        print(Fore.BLUE + "\nProducts related to Purchase History & Other Popular Items:" + Fore.WHITE)
        print(tabulate(pd_ret_df, headers = 'keys', tablefmt = 'rst'))

    def get_services(self, cust_index, month=CURMONTH, num_rows=50):
        ncf_ret = self.get_ncf_service_rec(cust_index)
        pd_ret = self.get_pd_service_rec(cust_index, month)
        if ncf_ret == None or pd_ret == None:
            print("There is no service purchase history information for customer %s, so unable to make recommendations!" % (cust_index))
            return

        ncf_ret_df = self.dfmerge135.df5[self.dfmerge135.df5['br_c'].isin(ncf_ret)]
        sorterIndex = dict(zip(ncf_ret, range(len(ncf_ret))))
        ncf_ret_df['rank'] = ncf_ret_df['br_c'].map(sorterIndex)
        ncf_ret_df = ncf_ret_df.sort_values(by='rank')
        ncf_ret_df = ncf_ret_df[['br_c']].reset_index(drop=True).head(num_rows)

        pd_ret_df = self.dfmerge135.df5[self.dfmerge135.df5['br_c'].isin(pd_ret)]
        sorterIndex = dict(zip(pd_ret, range(len(pd_ret))))
        pd_ret_df['rank'] = pd_ret_df['br_c'].map(sorterIndex)
        pd_ret_df = pd_ret_df.sort_values(by='rank')
        pd_ret_df = pd_ret_df[['br_c']].reset_index(drop=True).head(num_rows)

        print(Fore.CYAN + "\nTop Recommendable Services:" + Fore.WHITE)
        print(tabulate(ncf_ret_df, headers = 'keys', tablefmt = 'rst'))
        print(Fore.CYAN + "\nServices related to Purchase History & Other Popular Services:" + Fore.WHITE)
        print(tabulate(pd_ret_df, headers = 'keys', tablefmt = 'rst'))

if __name__ == "__main__":
    cfg = get_configs()
    recsys = RecSys(cfg)
    if cfg.customer_id == -1:
        print("Unable to make recommendations: missing customer ID!")
        exit()
    if cfg.recnum == -1 and cfg.month == -1:
        recsys.get_products(cfg.customer_id)
        recsys.get_services(cfg.customer_id)
    elif cfg.recnum == -1 and cfg.month != -1:
        recsys.get_products(cfg.customer_id, month=cfg.month)
        recsys.get_services(cfg.customer_id, month=cfg.month)
    elif cfg.recnum != -1 and cfg.month == -1:
        recsys.get_products(cfg.customer_id, num_rows=cfg.recnum)
        recsys.get_services(cfg.customer_id, num_rows=cfg.recnum)
    else:
        recsys.get_products(cfg.customer_id, month=cfg.month, num_rows=cfg.recnum)
        recsys.get_services(cfg.customer_id, month=cfg.month, num_rows=cfg.recnum)