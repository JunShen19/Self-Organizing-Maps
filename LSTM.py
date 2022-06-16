from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from tool import *
from matplotlib.pyplot import figure
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError


torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WaterLevelLSTM(nn.Module):
    # batch_first: true -> (batch, seq, feature: n_layers)
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(WaterLevelLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out       

def data_mean_df(ele_df, read_dictionary):
    ele_mean_df = pd.DataFrame()
    ele_df.columns = ele_df.columns.str.replace("\.0","")

    for i, v in enumerate(read_dictionary):
        if len(read_dictionary[v]) == 0:
            pass
        else:
            ele_mean_df[v] = ele_df.loc[:,read_dictionary[v].astype(str)].mean(axis=1).values
    return ele_mean_df

def data_preprocessing(ele_df, sc, sc_w, seq_length, num_classes, read_dictionary, set_name):
    dateList = timeRange('2007-01-01', '2018-06-30')
    startDate = datetime(2007, 1, 1, 0, 0)
    endDate = datetime(2018, 7, 1, 0, 0)
    
    dq_df = pd.read_csv(r'D:\20190103_地下水專案_處理\水利署地下水位資料\觀測井每天的平均地下水位_已過濾.csv', encoding='big5', index_col = 0)
    dq_df['date'] = dq_df['date'].astype('datetime64')
    dq_df = dq_df.iloc[dq_df.loc[dq_df.date == startDate].index[0]:dq_df.loc[dq_df.date == endDate].index[0],:]
    dq_df = dq_df.reset_index()
    del dq_df['index']
    well_name = 'HQ_九隆(3)'
    train_test_index = dq_df.loc[dq_df.date==datetime(2017, 1, 1, 0, 0)].index[0]

    date_index = dateList.index(startDate)
    if set_name == "all_mean" or 'waterlevel':
        pass
    else:
        ele_df.columns = ele_df.columns.str.replace("\.0","")

        ele_value = read_dictionary[set_name]
        ele_df = ele_df.loc[date_index:,ele_value.astype(str)]
        ele_df = ele_df.fillna(value=ele_df.mean(numeric_only=True))
    
    sc.fit(ele_df.iloc[:train_test_index,:])
    ele_df = pd.DataFrame(sc.transform(ele_df.iloc[:,:]))

    #　waterlevel
    sc_w.fit(dq_df.loc[:train_test_index,well_name].values.reshape(-1, 1))
    df = pd.DataFrame(sc_w.transform(dq_df.loc[:,well_name].values.reshape(-1, 1)))

    if set_name == "waterlevel":
        pass
    else:
        df = pd.concat([ele_df, df], axis=1)

    x, y = sliding_windows(df, seq_length, num_classes)

    del df
    del ele_df
    
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_test_index-seq_length])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_test_index-seq_length])))

    testX = Variable(torch.Tensor(np.array(x[train_test_index-seq_length:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_test_index-seq_length:len(y)])))

    print(dataX.size(), dataY.size())
    print(trainX.size(), trainY.size())
    print(testX.size(), testY.size())

    return dataX, dataY, trainX, trainY, testX, testY, sc, sc_w


def main():
    sc = MinMaxScaler()
    sc_w = MinMaxScaler()

    num_epochs = 400
    learning_rate = 0.005

    hidden_size = 256 
    num_layers = 1
    seq_length = 30
    num_classes = 60

    set_name = "all_mean"

    read_dictionary = np.load('som_ele_dict.npy',allow_pickle='TRUE').item()
    ele_df = read_ele_csv(r'D:\JunShen\dataset\彰化_雲林用電量(專用電表_day_raw.csv')

    ele_mean_df = data_mean_df(ele_df, read_dictionary)

    dataX, dataY, trainX, trainY, testX, testY, sc, sc_w= data_preprocessing(ele_mean_df, sc, sc_w, seq_length, num_classes, read_dictionary, set_name)

    input_size = dataX.shape[-1] # The number of expected features in the input x


    model = WaterLevelLSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(num_epochs):
        outputs = model(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY.view(outputs.size()))
        
        loss.backward()
        
        optimizer.step()

        # Testing the model
        with torch.no_grad():
            predicts = model(testX)
            val_loss = criterion(predicts, testY.view(predicts.size()))

        # eval
        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f, val_loss: %1.5f" % (epoch, loss.item(), val_loss.item()))

    model.eval()
    train_predict = model(dataX)

    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()

    data_predict = sc_w.inverse_transform(data_predict)
    dataY_plot = sc_w.inverse_transform(dataY_plot.reshape(data_predict.shape))
    
    data_predict60 = list(data_predict[0,:])
    for i in range(1,len(data_predict)):
        data_predict60.append(data_predict[i,-1])
    data_predict60 = np.array(data_predict60).flatten()

    dataY_plot60 = list(dataY_plot[0,:])
    for i in range(1,len(dataY_plot)):
        dataY_plot60.append(dataY_plot[i,-1])
    dataY_plot60 = np.array(dataY_plot60).flatten()

    data_predict_train = list(data_predict[0,:])
    for i in range(1,len(data_predict)-testX.shape[0]-num_classes+1):
        data_predict_train.append(data_predict[i,-1])
    data_predict_train = np.array(data_predict_train).flatten()
    data_predict_test = list(data_predict[trainX.shape[0],:])
    for i in range(trainX.shape[0]+1,len(data_predict)):
        data_predict_test.append(data_predict[i,-1])
    data_predict_test = np.array(data_predict_test).flatten()
    data_predict_test = np.concatenate([data_predict_train,data_predict_test])

    figure(figsize=(20, 16), dpi=80)
    plt.axvline(x=3653-seq_length, c='r', linestyle='--')
    plt.axvline(x=3653+seq_length, c='r', linestyle='--')

    plt.plot(dataY_plot60, color = 'r')
    plt.plot(data_predict_test, color = 'b')
    plt.plot(data_predict60, color = 'y')
    plt.suptitle('Time-Series Prediction')
    plt.show()

    # 單位不同(正規化已經反轉)
    mean_squared_error = MeanSquaredError()
    print(mean_squared_error(torch.tensor(data_predict[trainX.shape[0],:]), torch.tensor(dataY_plot[trainX.shape[0],:])))

if __name__ == "__main__":
    main()