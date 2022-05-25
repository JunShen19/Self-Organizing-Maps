import numpy as np
from datetime import datetime, timedelta 
import pandas as pd
import time

def readEleData(path):
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