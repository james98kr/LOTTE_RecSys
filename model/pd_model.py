import pandas as pd
pd.options.mode.chained_assignment = None

class DF_MERGE_124:
    def __init__(self, dfcust, dfpurchase, dfinfo):
        dfmerge = pd.merge(pd.merge(dfpurchase, dfcust), dfinfo)
        dfmerge['de_dt'] = dfmerge['de_dt'].astype(str)
        dfmerge['de_dt'] = pd.to_datetime(dfmerge['de_dt'])
        dfmerge['month'] = dfmerge['de_dt'].dt.month
        self.df1 = dfcust
        self.df2 = dfpurchase
        self.df4 = dfinfo
        self.dfmerge = dfmerge
        self.cust_dict2 = {c: i for i,c in enumerate(self.df2['cust'].unique())}
        self.inverse_cust_dict2 = {i: c for i,c in enumerate(self.df2['cust'].unique())}
        self.prod_dict =  {p: i for i,p in enumerate(self.df4['pd_c'].unique())}
        self.inverse_prod_dict =  {i: p for i,p in enumerate(self.df4['pd_c'].unique())}

    def getdfmerge(self):
        return self.dfmerge

    def getmonths(self, month):
        before = (month - 1) % 12 if month != 1 else 12
        after = (month + 1) % 12 if month != 11 else 12
        monthlist = [before, month, after]
        return monthlist

    # Based on the customer's purchase history within a given month, return products that are in the same category
    # as the ones that the customer bought most frequently
    def find_relevant_products(self, cust_id, month, prod_num=5):
        if isinstance(cust_id, int):
            cust_id = self.inverse_cust_dict2[cust_id]
        if cust_id not in self.cust_dict2:
            return None
        purchased = self.dfmerge[(self.dfmerge['cust'] == cust_id) & (self.dfmerge['month'].isin(self.getmonths(month)))]
        purchased = purchased.groupby(['pd_c', 'clac_mcls_nm'], as_index=False)['buy_ct'].sum().sort_values(by='buy_ct', ascending=False)
        purchased_category = list(purchased.head(prod_num)['clac_mcls_nm'])
        output = self.df4[self.df4['clac_mcls_nm'].isin(purchased_category)]
        sorterIndex = dict(zip(purchased_category, range(len(purchased_category))))
        output['clac_mcls_nm_rank'] = output['clac_mcls_nm'].map(sorterIndex)
        output = output.sort_values(by='clac_mcls_nm_rank')
        return output[['pd_c', 'pd_nm']]

    # Based on the customer's age, gender, and area of residence, return products that are bought most frequently
    # by the people in the same age/gender/residence group
    def find_top_products(self, cust_id, prod_num=20):
        if isinstance(cust_id, int):
            cust_id = self.inverse_cust_dict2[cust_id]
        if cust_id not in self.cust_dict2:
            return None
        personal_info = self.df1[self.df1['cust'] == cust_id]
        age = personal_info['ages'].to_string(index=False)
        gender = personal_info['ma_fem_dv'].to_string(index=False)
        area = personal_info['zon_hlv'].to_string(index=False)
        query = self.dfmerge[(self.dfmerge['ages'] == age) & (self.dfmerge['ma_fem_dv'] == gender) & (self.dfmerge['zon_hlv'] == area)]
        query = query.groupby(['pd_c', 'pd_nm'], as_index=False)['buy_ct'].sum().sort_values(by='buy_ct', ascending=False).head(prod_num)
        return query[['pd_c', 'pd_nm']]

class DF_MERGE_135:
    def __init__(self, dfcust, dfpurchase, dfinfo):
        dfmerge = pd.merge(dfpurchase, dfcust)
        dfmerge.rename(columns = {'zon_hlv':'cust_zon_hlv'}, inplace = True)
        dfmerge = pd.merge(dfmerge, dfinfo)
        dfmerge['de_dt'] = dfmerge['de_dt'].astype(str)
        dfmerge['de_dt'] = pd.to_datetime(dfmerge['de_dt'])
        dfmerge['month'] = dfmerge['de_dt'].dt.month
        self.df1 = dfcust
        self.df3 = dfpurchase
        self.df5 = dfinfo
        self.dfmerge = dfmerge
        self.cust_dict3 = {c: i for i,c in enumerate(self.df3['cust'].unique())}
        self.inverse_cust_dict3 = {i: c for i,c in enumerate(self.df3['cust'].unique())}
        self.serv_dict = {p: i for i,p in enumerate(self.df5['br_c'].unique())}
        self.inverse_serv_dict = {i: p for i,p in enumerate(self.df5['br_c'].unique())}

    def getdfmerge(self):
        return self.dfmerge

    def getmonths(self, month):
        before = (month - 1) % 12 if month != 1 else 12
        after = (month + 1) % 12 if month != 11 else 12
        monthlist = [before, month, after]
        return monthlist

    # Based on the customer's purchase history within a given month, return services that are in the same category
    # as the products that the customer bought most frequently, but prioritize services that are in the same region
    def find_relevant_servs(self, cust_id, month, serv_num=5):
        if isinstance(cust_id, int):
            cust_id = self.inverse_cust_dict3[cust_id]
        if cust_id not in self.cust_dict3:
            return None
        purchased = self.dfmerge[(self.dfmerge['cust'] == cust_id) & (self.dfmerge['month'].isin(self.getmonths(month)))]
        purchased = purchased.groupby(['cop_c', 'zon_mcls'], as_index=False).size().sort_values(by='size', ascending=False).head(serv_num)
        if purchased.empty:
            purchased = self.dfmerge[(self.dfmerge['cust'] == cust_id)]
            purchased = purchased.groupby(['cop_c', 'zon_mcls'], as_index=False).size().sort_values(by='size', ascending=False).head(serv_num)
        purchased = list(zip(purchased['cop_c'], purchased['zon_mcls']))
        total = self.dfmerge.groupby(['br_c', 'cop_c', 'zon_mcls'], as_index=False).size().sort_values(by='size', ascending=False).reset_index(drop=True)
        ret = []
        for category, zone in purchased:
            temp = list(total[(total['cop_c'] == category) & (total['zon_mcls'] == zone)]['br_c'])
            ret = ret + temp
        ret = pd.DataFrame(ret, columns=['br_c'])
        return ret

    # Based on the customer's age, gender, and area of residence, return services that are bought most frequently
    # by the people in the same age/gender/residence group
    def find_top_servs(self, cust_id, serv_num=20):
        if isinstance(cust_id, int):
            cust_id = self.inverse_cust_dict3[cust_id]
        if cust_id not in self.cust_dict3:
            return None
        personal_info = self.df1[self.df1['cust'] == cust_id]
        age = personal_info['ages'].to_string(index=False)
        gender = personal_info['ma_fem_dv'].to_string(index=False)
        area = personal_info['zon_hlv'].to_string(index=False)
        query = self.dfmerge[(self.dfmerge['ages'] == age) & (self.dfmerge['ma_fem_dv'] == gender) & (self.dfmerge['cust_zon_hlv'] == area)]
        query = query.groupby(['br_c'], as_index=False)['buy_am'].sum().sort_values(by='buy_am', ascending=False).head(serv_num)
        return query[['br_c']]