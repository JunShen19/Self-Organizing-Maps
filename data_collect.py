import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tool import *

startDate = datetime(2007, 1, 1, 0, 0)
endDate = datetime(2018, 7, 1, 0, 0)
well_name = '線西(4)'
    
dq_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+'九隆(3)'+'\\'+'feature csv'+'\\'+f'waterlevel.csv', encoding='big5', index_col=0, parse_dates=['date'])

ele_df = read_ele_csv(r'D:\JunShen\dataset\彰化_雲林用電量(專用電表_day_raw.csv')
dateList = timeRange('2007-01-01', '2018-06-30')

# deal cluster feature
for cluster_num in [4,9,16,25]:
    # load elename value
    read_dictionary = np.load(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'som_ele_dict_' + f'{cluster_num}.npy',allow_pickle='TRUE').item()

    ele_mean_df = pd.DataFrame()
    ele_df.columns = ele_df.columns.str.replace("\.0","")

    for i, v in enumerate(read_dictionary):
        if len(read_dictionary[v]) == 0:
            pass
        else:
            ele_mean_df[v] = ele_df.loc[:,read_dictionary[v].astype(str)].mean(axis=1).values
    
    ele_mean_df.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'feature csv'+'\\'+f'som_feature_{cluster_num}cluster.csv')

# deal cluster feature with move average
for cluster_num in [4,9,16,25]:
    # load elename value
    read_dictionary = np.load(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'som_ele_dict_' + f'{cluster_num}.npy',allow_pickle='TRUE').item()

    ele_mean_df = pd.DataFrame()
    ele_df.columns = ele_df.columns.str.replace("\.0","")

    for i, v in enumerate(read_dictionary):
        if len(read_dictionary[v]) == 0:
            pass
        else:
            ele_mean_df[v] = ele_df.loc[:,read_dictionary[v].astype(str)].mean(axis=1).values
    
    ori_col = ele_mean_df.columns
    for i in ori_col:
        ele_mean_df[i+'SMA60'] = ele_mean_df[i].rolling(60).mean()

        ele_mean_df[i+'SMA60'].fillna(method='ffill', inplace=True)
        ele_mean_df[i+'SMA60'].fillna(method='bfill', inplace=True)

    ele_mean_df.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'feature csv'+'\\'+f'som_feature_{cluster_num}cluster_SMA.csv')

# deal 5km feature
# load elename value
read_dictionary = np.load(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'5km_ele_dict.npy',allow_pickle='TRUE').item()

ele_df.columns = ele_df.columns.str.replace("\.0","")

km_ele_df = pd.DataFrame(ele_df.loc[:,read_dictionary['all'].astype(str)].values)

km_ele_df.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+well_name+'\\'+'feature csv'+'\\'+f'5km_ele_all.csv')
