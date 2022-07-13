from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tool import *
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#test
true = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10]])
pred = np.array([[0.2,1.2,2,3,3.7,5,6.2,6.8,7.5,8.5],[0.2,1.2,2,3,3.7,5,6.2,6.8,7.5,8.5]])
true = [0,1,2,3,4,5,6,7,8,9,10]

def eva_nth_performance(y_true_total, y_pred_total, cluster_num, writer, exp_name, fold):
    df_out = pd.DataFrame()
    y_true_total = y_true_total.transpose()
    y_pred_total = y_pred_total.transpose()
    np_out = []
    for nth in range(y_true_total.shape[0]):
        print(f"{nth+1}th day")
        r2 = r2_score(y_true_total[nth],y_pred_total[nth])
        print("ACC",ACC(y_true_total[nth],y_pred_total[nth], 0.1))
        np_out.append(ACC(y_true_total[nth],y_pred_total[nth], 0.1))
        print("R2",r2.round(3))
        np_out.append(r2.round(3))
        print("MAE",mean_absolute_error(y_true_total[nth],y_pred_total[nth]))
        np_out.append(mean_absolute_error(y_true_total[nth],y_pred_total[nth]))
        print("MSE",mean_squared_error(y_true_total[nth],y_pred_total[nth]))
        np_out.append(mean_squared_error(y_true_total[nth],y_pred_total[nth]))
        print("RMSE",np.sqrt(mean_squared_error(y_true_total[nth],y_pred_total[nth])))
        np_out.append(np.sqrt(mean_squared_error(y_true_total[nth],y_pred_total[nth])))
        print("--------------")
        df_out[f'{nth+1}th day'] = np.array(np_out)
        writer.add_scalar('ACC_nth', ACC(y_true_total[nth],y_pred_total[nth], 0.1), nth+1)
        writer.add_scalar('R2_nth', r2.round(3), nth+1)
        writer.add_scalar('MAE_nth', mean_absolute_error(y_true_total[nth],y_pred_total[nth]), nth+1)
        writer.add_scalar('MSE_nth', mean_squared_error(y_true_total[nth],y_pred_total[nth]), nth+1)
        writer.add_scalar('RMSE_nth', np.sqrt(mean_squared_error(y_true_total[nth],y_pred_total[nth])), nth+1)
        np_out = []
    df_out = df_out.set_index(pd.Series(['ACC','R2','MAE','MSE','RMSE']))
    df_out.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature\Jiulong(3)\result evaluation'+ '\\' + f'eva_nth_{exp_name}_fold{fold+1}_performance_cluster_{cluster_num}.csv')


def create_hill_marker(y_total): 
    hill_marker = []       
    for i in range(len(y_total)-1):
        if y_total[i+1] - y_total[i] > 0:
            marker = 1
        else:
            marker = -1
        hill_marker.append(marker)
    return np.array(hill_marker)

# hill_marker: 1darray
def eva_hill(y_true_total_1d, y_pred_total_1d, hill_marker, cluster_num, exp_name,fold):
    df_out = pd.DataFrame()
    np_out = []

    uphill = np.where(hill_marker==1, y_true_total_1d[:-1], np.nan)
    uphill = uphill[~np.isnan(uphill)]
    uphill = uphill.astype(np.int16) 

    downhill = np.where(hill_marker==-1, y_true_total_1d[:-1], np.nan)
    downhill = downhill[~np.isnan(downhill)]
    downhill = downhill.astype(np.int16)

    uphill_true = y_true_total_1d[uphill]
    downhill_true = y_true_total_1d[downhill]

    uphill_pred = y_pred_total_1d[uphill]
    downhill_pred = y_pred_total_1d[downhill]

    print("uphill:")
    r2 = r2_score(uphill_true,uphill_pred)
    print("ACC",ACC(uphill_true, uphill_pred, 0.1))
    np_out.append(ACC(uphill_true, uphill_pred, 0.1))
    print("R2",r2.round(3))
    np_out.append(r2.round(3))
    print("MAE",mean_absolute_error(uphill_true,uphill_pred))
    np_out.append(mean_absolute_error(uphill_true,uphill_pred))
    print("MSE",mean_squared_error(uphill_true,uphill_pred))
    np_out.append(mean_squared_error(uphill_true,uphill_pred))
    print("RMSE",np.sqrt(mean_squared_error(uphill_true,uphill_pred)))
    np_out.append(np.sqrt(mean_squared_error(uphill_true,uphill_pred)))
    print("--------------")
    df_out['uphill'] = np.array(np_out)

    np_out = []

    print("downhill:")
    r2 = r2_score(downhill_true,downhill_pred)
    print("ACC",ACC(downhill_true, downhill_pred, 0.1))
    np_out.append(ACC(downhill_true, downhill_pred, 0.1))
    print("R2",r2.round(3))
    np_out.append(r2.round(3))
    print("MAE",mean_absolute_error(downhill_true,downhill_pred))
    np_out.append(mean_absolute_error(downhill_true,downhill_pred))
    print("MSE",mean_squared_error(downhill_true,downhill_pred))
    np_out.append(mean_squared_error(downhill_true,downhill_pred))
    print("RMSE",np.sqrt(mean_squared_error(downhill_true,downhill_pred)))
    np_out.append(np.sqrt(mean_squared_error(downhill_true,downhill_pred)))
    print("--------------")
    df_out['downhill'] = np.array(np_out)
    df_out = df_out.set_index(pd.Series(['ACC','R2','MAE','MSE','RMSE']))
    df_out.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature\Jiulong(3)\result evaluation' +'\\'+ f'eva_hill_{exp_name}_fold{fold+1}_performance_cluster_{cluster_num}.csv')

def eva_60days_performance(y_true_total, y_pred_total, cluster_num, exp_name, fold):
    df_out = pd.DataFrame()
    np_out = []

    print("Total Performance")
    r2 = r2_score(y_true_total,y_pred_total)
    print("R2",r2.round(3))
    np_out.append(r2.round(3))
    print("MAE",mean_absolute_error(y_true_total,y_pred_total))
    np_out.append(mean_absolute_error(y_true_total,y_pred_total))
    print("MSE",mean_squared_error(y_true_total,y_pred_total))
    np_out.append(mean_squared_error(y_true_total,y_pred_total))
    print("RMSE",np.sqrt(mean_squared_error(y_true_total,y_pred_total)))
    np_out.append(np.sqrt(mean_squared_error(y_true_total,y_pred_total)))
    print("--------------")
    df_out['eva_60days_performance'] = np.array(np_out)
    df_out = df_out.set_index(pd.Series(['R2','MAE','MSE','RMSE']))
    df_out.to_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature\Jiulong(3)\result evaluation' +'\\'+ f'eva_{exp_name}_fold{fold+1}_performance_cluster_{cluster_num}.csv')
    
    return np.array(np_out)