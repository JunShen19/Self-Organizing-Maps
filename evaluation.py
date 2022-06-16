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

def eva_nth_performance(y_true_total, y_pred_total):
    y_true_total = y_true_total.transpose()
    y_pred_total = y_pred_total.transpose()
    for nth in range(y_true_total.shape[0]):
        print(f"{nth+1} day")
        r2 = r2_score(y_true_total[nth],y_pred_total[nth])
        print("R2",r2.round(3))
        print("MAE",mean_absolute_error(y_true_total[nth],y_pred_total[nth]))
        print("MSE",mean_squared_error(y_true_total[nth],y_pred_total[nth]))
        print("RMSE",np.sqrt(mean_squared_error(y_true_total[nth],y_pred_total[nth])))
        print("--------------")

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
def eva_hill(y_true_total_1d, y_pred_total_1d, hill_marker):
    uphill = np.where(hill_marker==1, true[:-1], np.nan)
    uphill = uphill[~np.isnan(uphill)]

    downhill = np.where(hill_marker==-1, true[:-1], np.nan)
    downhill = downhill[~np.isnan(downhill)]

    uphill_true = y_true_total_1d[uphill]
    downhill_true = y_true_total_1d[downhill]

    uphill_pred = y_pred_total_1d[uphill]
    downhill_pred = y_pred_total_1d[downhill]