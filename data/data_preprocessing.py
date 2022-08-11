import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# In order to create the customer-product matrix from scratch, set CREATE_CUSTOMER_PRODUCT_MATRIX to True
def create_cust_prod_matrix(CREATE_CUSTOMER_PRODUCT_MATRIX, path_to_data):
    path_to_data = path_to_data + 'data/'
    if CREATE_CUSTOMER_PRODUCT_MATRIX == True:
        # Read csv files to pandas dataframe
        df2 = pd.read_csv(path_to_data + 'raw/LPOINT_BIG_COMP_02_PDDE.csv')
        df4 = pd.read_csv(path_to_data + 'raw/LPOINT_BIG_COMP_04_PD_CLAC.csv')

        # Fill NaN with empty string for df3, and create required dictionaries
        result = df2['cust'].unique()
        cust_dict2 = {c: i for i,c in enumerate(result)}
        result = df4['pd_c'].unique()
        prod_dict = {p: i for i,p in enumerate(result)}
        inverse_prod_dict = {i: p for i,p in enumerate(result)}

        sampling = 4
        matrix = df2[['cust', 'pd_c']].to_numpy()
        group_by_user = dict(df2[['cust', 'pd_c']].groupby('cust')['pd_c'].apply(' '.join))
        cust_prod_matrix = np.zeros((len(matrix) * sampling, 3))
        
        for i, row in enumerate(matrix):
            cust_prod_matrix[sampling*i][0] = cust_dict2[row[0]]
            cust_prod_matrix[sampling*i][1] = prod_dict[row[1]]
            cust_prod_matrix[sampling*i][2] = float(1)
            string_to_compare = group_by_user[row[0]]
            temp = []
            while len(temp) != (sampling - 1):
                randprod = random.randint(0, 1932)
                randprod_str = inverse_prod_dict[randprod]
                if randprod_str not in string_to_compare:
                    temp.append(randprod)
            for j in range(1,sampling):
                cust_prod_matrix[sampling*i + j][0] = cust_dict2[row[0]]
                cust_prod_matrix[sampling*i + j][1] = temp[j - 1]
                cust_prod_matrix[sampling*i + j][2] = float(0)
        np.save(path_to_data + 'matrix/cust_prod_matrix.npy', cust_prod_matrix)
    else:
        cust_prod_matrix = np.load(path_to_data + 'matrix/cust_prod_matrix.npy')
    return cust_prod_matrix

# In order to create the customer-service matrix from scratch, set CREATE_CUSTOMER_SERVICE_MATRIX to True
def create_cust_serv_matrix(CREATE_CUSTOMER_SERVICE_MATRIX, path_to_data):
    path_to_data = path_to_data + 'data/'
    if CREATE_CUSTOMER_SERVICE_MATRIX == True:
        # Read csv files to pandas dataframe
        df3 = pd.read_csv(path_to_data + 'raw/LPOINT_BIG_COMP_03_COP_U.csv')
        df5 = pd.read_csv(path_to_data + 'raw/LPOINT_BIG_COMP_05_BR.csv')

        # Fill NaN with empty string for df3, and create required dictionaries
        df3.fillna('', inplace=True)
        df3.drop(df3[df3['br_c'] == ''].index, inplace=True)
        result = df3['cust'].unique()
        cust_dict3 = {c: i for i,c in enumerate(result)}
        result = df5['br_c'].unique()
        serv_dict = {p: i for i,p in enumerate(result)}
        inverse_serv_dict = {i: p for i,p in enumerate(result)}

        sampling = 4
        matrix = df3[['cust', 'br_c']].to_numpy()
        group_by_user = dict(df3[['cust', 'br_c']].groupby('cust')['br_c'].apply(' '.join))
        cust_serv_matrix = np.zeros((len(matrix) * sampling, 3))
        for i, row in enumerate(matrix):
            cust_serv_matrix[sampling*i][0] = cust_dict3[row[0]]
            cust_serv_matrix[sampling*i][1] = serv_dict[row[1]]
            cust_serv_matrix[sampling*i][2] = float(1)
            string_to_compare = group_by_user[row[0]]
            temp = []
            while len(temp) != (sampling - 1):
                randserv = random.randint(0, 8807)
                randserv_str = inverse_serv_dict[randserv]
                if randserv_str not in string_to_compare:
                    temp.append(randserv)
            for j in range(1,sampling):
                cust_serv_matrix[sampling*i + j][0] = cust_dict3[row[0]]
                cust_serv_matrix[sampling*i + j][1] = temp[j - 1]
                cust_serv_matrix[sampling*i + j][2] = float(0)
        np.save(path_to_data + 'matrix/cust_serv_matrix.npy', cust_serv_matrix)
    else:
        cust_serv_matrix = np.load(path_to_data + 'matrix/cust_serv_matrix.npy')
    return cust_serv_matrix

class MyDataSet(Dataset):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    def __len__(self):
        return self.matrix.shape[0]
    
    def __getitem__(self, index):
        cust = torch.Tensor([self.matrix[index][0]]).long().to(self.device)
        item = torch.Tensor([self.matrix[index][1]]).long().to(self.device)
        label = torch.Tensor([self.matrix[index][2]]).long().to(self.device)
        return (cust, item), label

if __name__ == "__main__":
    ROOT_PATH = '/home/james/lotte/'
    create_cust_prod_matrix(True, ROOT_PATH)
    create_cust_serv_matrix(True, ROOT_PATH)