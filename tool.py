import numpy as np
from datetime import datetime, timedelta 
import pandas as pd
import time

def read_ele_csv(path):
    reader = pd.read_csv(path, iterator=True, low_memory=False , encoding = 'big5', index_col=0)

    chunksize = 500
    state = True
    chunks = []
    c = 0
    print('讀取用電資料:', path)
    stime = time.time()
    while state:
        try:
            chunks.append(reader.get_chunk(chunksize))
            t = time.time() - stime
            c += 1
            print('讀取', c * chunksize, '筆資料之花費時間:', round(t/60, 2), '分鐘')
        except StopIteration:
            state = False
            print('Stop iteration!')

    t = time.time() - stime
    df_ele = pd.concat(chunks, ignore_index=True)
    print('串接資料花費時間:', round(t/60, 2), '分鐘')
    print('用電資料讀取完成!!')
    
    return df_ele

def get_col_by_radius(ele_df, df_hq68well, distance, wellName):
        well_x_y = df_hq68well.loc[df_hq68well.name_c == wellName, ['TWD97TM2(x坐標)','TWD97TM2(y坐標)']].values[0]
        center = well_x_y

        con = ((ele_df.TWD97_X-center[0])**2 + (ele_df.TWD97_Y-center[1])**2) ** (1/2) < distance
        columns = []
        for i in ele_df.loc[con,"修正電號"].values:
            columns.append(str(i))
        print('所取之電表數:',len(columns))     
        
        return columns

# 準確率計算
def ACC(test, pred, th):
    right = 0
    min_test = min(test)
    max_test = max(test)
    th = (max_test - min_test) * th    
    for i in range(len(test)):
        if (test[i] - th) < pred[i] < (test[i] + th):
            right += 1

    return right / len(test)

def preprocessing_ele_df(ele_df):
    stime = time.time()
    print('調整前ele_df.shape:', ele_df.shape)
    ele_df['mean'] = ele_df.iloc[:,:-2].loc[:,"9601":].mean(axis=1)
    
    center = [191335, 2628306]
    #Case1:
    condition = ((ele_df.TWD97_X-center[0])**2 + (ele_df.TWD97_Y-center[1])**2) ** (1/2) > 300000
    print('✔ 距離問題(超過300km):', len(ele_df.loc[condition].index))
    df_after_process = ele_df.drop(ele_df.loc[condition].index)
    
    #Case2:
    print('✔ No data(mean):', len(df_after_process.loc[df_after_process['mean'].isnull()].index))
    df_after_process = df_after_process.drop(df_after_process.loc[df_after_process['mean'].isnull()].index)
    
    #Case3:
    print('✔ Using avg(mean) == 0:', len(df_after_process.loc[df_after_process['mean'] == 0].index))
    df_after_process = df_after_process.drop(df_after_process.loc[df_after_process['mean'] == 0].index)
    
    #Case4:
    print('✔ Using avg(mean) > 1000:', len(df_after_process.loc[df_after_process['mean'] > 1000].index))
    df_after_process = df_after_process.drop(df_after_process.loc[df_after_process['mean'] > 1000].index)
    
    return df_after_process

def timeRange(start: str, end: str) ->list:
    timeList = []
    oneDay = timedelta(days=1)
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    for i in range((end-start).days + 1):
        timeList.append(start + oneDay * i)
    return timeList

def well_df(df_hqInf, df_hqInf2, wellNum):
    wells = df_hqInf.loc[df_hqInf.num_well==wellNum, 'wells'].values[0].split(';')
    df_hq68well = pd.DataFrame(columns = df_hqInf2.columns)
    for well in range(len(wells)):
        df_hq68well.loc[well,:] = df_hqInf2.loc[df_hqInf2.name_c == wells[well][3:],:].values
    df_hq68well.elev_cur = df_hq68well.elev_cur.astype(np.float32)
    
    return df_hq68well

def sliding_windows(df, seq_length, pred_length):
    x = []
    y = []

    for i in range(len(df)-seq_length-pred_length-1):
        _x = df.iloc[i:(i+seq_length),:]
        _y = df.iloc[i+seq_length:i+seq_length+pred_length,-1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


