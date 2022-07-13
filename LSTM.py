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
from evaluation import *
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter 
import math


torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init
dateList = timeRange('2007-01-01', '2018-06-30')
startDate = datetime(2007, 1, 1, 0, 0)
endDate = datetime(2018, 7, 1, 0, 0)

num_epochs = 500
learning_rate = 0.001
hidden_size = 256 
num_layers = 1 # 2->multiple
seq_length = 30
num_classes = 60
batch_size = 30
model_type = 'LSTM'

acc = [np.array(()) for _ in range(num_classes)]
mse = [np.array(()) for _ in range(num_classes)]
rmse = [np.array(()) for _ in range(num_classes)]
mae = [np.array(()) for _ in range(num_classes)]
r2 = [np.array(()) for _ in range(num_classes)]

result_df = pd.DataFrame()

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, seq_len=30, predict_len=60):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.transform = MinMaxScaler()


    def __len__(self):
        return self.X.__len__() - (self.seq_len+self.predict_len)

    def __getitem__(self, index):
        x_scaler = self.transform
        x_minmax = x_scaler.fit_transform(self.X)

        y_scaler = self.transform
        y_minmax = y_scaler.fit_transform(self.y.reshape(-1,1)).squeeze(-1)
        return (x_minmax[index:index+self.seq_len], y_minmax[index+self.seq_len:index+self.seq_len+self.predict_len])

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

        self.relu = nn.ReLU()


    def forward(self, x):
        # print(x.size())
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        out = self.relu(h_out)
        out = self.fc(out)
        
        return out       

def train_test_dataset(dataset, train_test_index_set):
    datasets = {}
    for k in range(1,6):
        train_idx = [i for i in range(train_test_index_set[k-1]-seq_length+1)]
        val_idx = [i for i in range(train_idx[-1]+1, len(dataset))]

        datasets[f'train{k}'] = Subset(dataset, train_idx)
        datasets[f'test{k}'] = Subset(dataset, val_idx)
    return datasets

def train_val_dataset(datasets, val_size=0.2):
    datasets_train = {}
    for k in range(1,6):
        datasets_train[f'train{k}'] = datasets[f'train{k}']

    for k in range(1,6):
        train_idx, val_idx = train_test_split([i for i in range(len(datasets[f'train{k}']))], test_size=val_size, shuffle=False)
        datasets[f'train{k}'] = Subset(datasets_train[f'train{k}'], train_idx)
        datasets[f'val{k}'] = Subset(datasets_train[f'train{k}'], val_idx)
    return datasets

def data_preprocessing(df):
    train_test_index_set = []
    train_test_index_set.append(df.loc[df.date==datetime(2017, 1, 1, 0, 0)].index[0])
    train_test_index_set.append(df.loc[df.date==datetime(2017, 3, 1, 0, 0)].index[0])
    train_test_index_set.append(df.loc[df.date==datetime(2017, 5, 1, 0, 0)].index[0])
    train_test_index_set.append(df.loc[df.date==datetime(2017, 7, 1, 0, 0)].index[0])
    train_test_index_set.append(df.loc[df.date==datetime(2017, 9, 1, 0, 0)].index[0])
    train_test_index_set.append(df.loc[df.date==datetime(2017, 11, 1, 0, 0)].index[0])

    date_col = df['date']
    del df['date']

    x = df.iloc[:,1:].values
    y = df.loc[:,well_name].values # waterlevel

    dataset = TimeseriesDataset(x, y, seq_length, num_classes)
    datasets = train_test_dataset(dataset, train_test_index_set)
    datasets = train_val_dataset(datasets, val_size=0.2)

    dataloaders = {x:DataLoader(datasets[x], batch_size, shuffle=False) for x in list(datasets.keys()) if 'test' not in x}
    testloaders = {x:DataLoader(datasets[x], batch_size, shuffle=False) for x in list(datasets.keys()) if 'test' in x}

    return dataloaders, testloaders

def test(device, model, test_loader, loss_function, sc, writer, k):
    # Settings
    model.eval()
    total = 0
    correct = 0
    
    outputs_data = [np.array(()) for _ in range(num_classes)]
    labels_data = [np.array(()) for _ in range(num_classes)]

    times = 0
    loss_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(inputs.to(torch.float32))
            loss = loss_function(outputs, labels.to(torch.float32))
            loss_total += loss.item()

            for i in range(labels.size(1)):
                outputs_data[i] = np.append(outputs_data[i], outputs.permute(1,0)[i].numpy())
                labels_data[i] = np.append(labels_data[i], labels.permute(1,0)[i].numpy())
                total += labels.size(0) #60x(4199-30)

                if times == 0:
                    result_df[f"fold-{k}_pred"] = outputs[i]
                    result_df[f"fold-{k}_true"] = labels[i]
                    times = 1
                else:
                    pass
    
    for i in range(len(outputs_data)):
        test = sc.inverse_transform(labels_data[i].reshape(-1, 1)).flatten()
        pred = sc.inverse_transform(outputs_data[i].reshape(-1, 1)).flatten()
        print(f'{i+1}th day:')
        print('Accuracy:', (abs(pred-test) < 0.01).sum() / pred.shape[0])
        print('MSE:', mean_squared_error(test, pred))
        print('RMSE:', math.sqrt(mean_squared_error(test, pred)))
        print('MAE:', mean_absolute_error(test, pred))
        print('r2:', r2_score(test, pred))
        acc[i] = np.append(acc[i],  (abs(pred-test) < 0.01).sum() / pred.shape[0])
        r2[i] = np.append(r2[i],   r2_score(test, pred))
        mae[i] = np.append(mae[i],  mean_absolute_error(test, pred))
        mse[i] = np.append(mse[i],  mean_squared_error(test, pred))
        rmse[i] = np.append(rmse[i],  math.sqrt(mean_squared_error(test, pred)))
        writer.add_scalar('ACC_nth', (abs(pred-test) < 0.01).sum() / pred.shape[0], i+1)
        writer.add_scalar('R2_nth', r2_score(test, pred), i+1)
        writer.add_scalar('MAE_nth', mean_absolute_error(test, pred), i+1)
        writer.add_scalar('MSE_nth', mean_squared_error(test, pred), i+1)
        writer.add_scalar('RMSE_nth', math.sqrt(mean_squared_error(test, pred)), i+1)
    
    outputs_total_flatten = torch.FloatTensor(outputs_data).flatten().numpy()
    labels_total_flatten = torch.FloatTensor(labels_data).flatten().numpy()
    
    outputs_total_flatten = sc.inverse_transform(outputs_total_flatten.reshape(-1, 1))
    labels_total_flatten = sc.inverse_transform(labels_total_flatten.reshape(-1, 1))

    # total
    correct = (abs(outputs_total_flatten - labels_total_flatten) < 0.01).sum()
    print('total:')
    print('Accurecy:', correct / total)
    print('MSE:', mean_squared_error(labels_total_flatten, outputs_total_flatten))
    print('RMSE:', math.sqrt(mean_squared_error(labels_total_flatten, outputs_total_flatten)))
    print('MAE:', mean_absolute_error(labels_total_flatten, outputs_total_flatten))
    print('r2:', r2_score(labels_total_flatten, outputs_total_flatten))
    
    writer.add_scalar('Accurecy:', correct / total, 1)
    writer.add_scalar('r2:', r2_score(labels_total_flatten, outputs_total_flatten), +1)
    writer.add_scalar('MAE:', mean_absolute_error(labels_total_flatten, outputs_total_flatten), 1)
    writer.add_scalar('MSE:', mean_squared_error(labels_total_flatten, outputs_total_flatten), 1)
    writer.add_scalar('RMSE:', math.sqrt(mean_squared_error(labels_total_flatten, outputs_total_flatten)), 1) 

    # test_loss
    
    print('test_loss:', loss_total / len(test_loader))
    writer.add_scalar("test_loss:", loss_total / len(test_loader), 1)
    
    return 

def validation(model, device, valid_loader, loss_function):
    # Settings
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for data in valid_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs.to(torch.float32))
            loss = loss_function(outputs, labels.to(torch.float32))
            loss_total += loss.item()

    return loss_total / len(valid_loader)

def train(device, model, epochs, optimizer, loss_function, train_loader, valid_loader, writer):
    # Early stopping
    the_last_loss = 100
    patience = 4
    trigger_times = 0

    for epoch in range(1, epochs+1):
        model.train()

        for times, data in enumerate(train_loader, 1):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(inputs.to(torch.float32))
            loss = loss_function(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            # Show progress
            if times % 100 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, times, len(train_loader), loss.item()))

        
        # Early stopping
        the_current_loss = validation(model, device, valid_loader, loss_function)
        print('The current loss:', the_current_loss)

        # add_writer
        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('val_loss', the_current_loss, epoch)

        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model

        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_loss = the_current_loss

    return model

def plotting(model, dataloaders, testloaders, k, sc, writer):
    model.eval()
    # plot y_true
    train_outputs_total = np.array(())
    with torch.no_grad():
        for i, data in enumerate(dataloaders[f'train{k}']):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(inputs.to(torch.float32))

            train_outputs_total = np.append(train_outputs_total, outputs.permute(1,0)[0].numpy())


    train_outputs_total = np.array(train_outputs_total, dtype='float').flatten()
    train_outputs_total = sc.inverse_transform(train_outputs_total.reshape(-1, 1))

    val_outputs_total = np.array(())
    with torch.no_grad():
        for i, data in enumerate(dataloaders[f'val{k}']):
            inputs = data[0].to(device)
            outputs = model(inputs.to(torch.float32))

            val_outputs_total = np.append(val_outputs_total, outputs.permute(1,0)[0].numpy())

    val_outputs_total = np.array(val_outputs_total, dtype='float').flatten()
    val_outputs_total = sc.inverse_transform(val_outputs_total.reshape(-1, 1))
    
    test_outputs_total = np.array(())
    with torch.no_grad():
        for i, data in enumerate(testloaders[f'test{k}']):
            inputs = data[0].to(device)
            outputs = model(inputs.to(torch.float32))

            if i == len(testloaders[f'test{k}'])-1:
                test_outputs_total = np.append(test_outputs_total, outputs.permute(1,0)[0][:-1].numpy())
                test_outputs_total = np.append(test_outputs_total, outputs[-1].numpy())
            else:
                test_outputs_total = np.append(test_outputs_total, outputs.permute(1,0)[0].numpy())

    test_outputs_total = np.array(test_outputs_total, dtype='float').flatten()
    test_outputs_total = sc.inverse_transform(test_outputs_total.reshape(-1, 1))
    

    # fit size
    y_true = df.loc[:,well_name].values

    y_train = [np.nan for _ in range(len(dateList))]
    for i in range(len(train_outputs_total)):
        y_train[seq_length+i] = train_outputs_total[i]

    y_val = [np.nan for _ in range(len(dateList))]
    for i in range(len(val_outputs_total)):
        y_val[seq_length+len(train_outputs_total)+i] = val_outputs_total[i]

    y_test = [np.nan for _ in range(len(dateList))]
    for i in range(len(test_outputs_total)):
        y_test[-len(test_outputs_total)+i] = test_outputs_total[i]

    fig, ax = plt.subplots(figsize=(20, 16), dpi=80)

    ax.set_xlabel("DATE")
    ax.set_ylabel("GroundWater Level")
    ax.set_title(f"{well_name}_{k}-fold",  fontproperties="SimSun")

    ax.plot(dateList, y_true, color = 'r', label='Measurements')
    ax.plot(dateList, y_train, color = 'g', label='Measurements')
    ax.plot(dateList, y_val, color = 'b', label='Measurements')
    ax.plot(dateList, y_test, color = 'black', label='Measurements')
    writer.add_figure('result',fig)

   
def main(df, model_type, well_name, set_num):

    dataloaders, testloaders= data_preprocessing(df)
    x,y = next(iter(testloaders['test1']))
    print(x.shape, y.shape)
    x,y = next(iter(dataloaders['train1']))
    print(x.shape, y.shape)

    input_size = x.shape[-1] # The number of expected features in the input x

    for k in range(1,6):
        exp_name = f'set-{set_num}_{k}-fold_model-{model_type}_{well_name}'
        writer = SummaryWriter('runs2'+'//'+exp_name)
        # train model

        if model_type == 'LSTM':
            model = WaterLevelLSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
        if model_type == 'Mutli-LSTM':
            model = WaterLevelLSTM(num_classes, input_size, hidden_size, 2, seq_length)

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model = train(device, model, num_epochs, optimizer, criterion, dataloaders[f'train{k}'], dataloaders[f'val{k}'], writer)

        sc = MinMaxScaler().fit(df.loc[:len(dataloaders[f'train{k}'].dataset), well_name].values.reshape(-1,1))
        
        test(device, model, testloaders[f'test{k}'], criterion, sc, writer, k)
        plotting(model, dataloaders, testloaders, k, sc, writer)

    eva = pd.DataFrame()
    print('average:')
    acc_day = []
    mse_day = []
    mae_day = []
    rmse_day = []
    r2_day = []

    for i in range(num_classes):
        acc_day.append(sum(acc[i])/len(acc[i]))
        mse_day.append(sum(mse[i])/len(mse[i]))
        mae_day.append(sum(mae[i])/len(mae[i]))
        rmse_day.append(sum(rmse[i])/len(rmse[i]))
        r2_day.append(sum(r2[i])/len(r2[i]))

    eva["ACC"] = np.array(acc_day)
    eva["MSE"] = np.array(mse_day)
    eva["RMSE"] = np.array(rmse_day)
    eva["MAE"] = np.array(mae_day)
    eva["R2"] = np.array(r2_day)

    eva.to_csv(r'D:\JunShen\SOM\result' + "\\" + f'eva_{well_name}_{model_type}_set{set_num}.csv')
    result_df.to_csv(r'D:\JunShen\SOM\result' + "\\" + f'result_{well_name}_{model_type}_set{set_num}.csv')
    
    #save model 
    torch.save(model, r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'model save'+'\\'+f'eva_{well_name}_{model_type}_set{set_num}.pt')

if __name__ == "__main__":
    font = {'weight' : 'normal',
        'size'   : 22}

    plt.rc('font', **font)

    dq_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature\九隆(3)\feature csv\waterlevel.csv', encoding='big5', index_col=0, parse_dates=['date'])
    # cluster_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_9cluster.csv', index_col=0)

    well_name_set = ['HQ_九隆(3)']
    model_type_set = ['LSTM']# or Multi-LSTM

    for model_type in model_type_set:
        for well_name in well_name_set:
            
            dq_well_df = dq_df.loc[:,['date', well_name]]
            if True in  dq_well_df.iloc[:,1].isnull():
                dq_well_df.iloc[:,1] = dq_well_df.iloc[:,1].interpolate(method='polynomial', order=2)

            for set_num in [6]:
                if set_num == 0:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_4cluster.csv', index_col=0)
                if set_num == 1:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_9cluster.csv', index_col=0)
                if set_num == 2:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_16cluster.csv', index_col=0)
                if set_num == 3:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_25cluster.csv', index_col=0)
                if set_num == 4:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_25cluster.csv', index_col=0)
                    month_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\month_cluster_by_ele.csv', index_col=0)
                    #convert column to datetimes and then to first day of month
                    month_df['month'] = pd.to_datetime(month_df['month'], format='%Y-%m')
                    month_df['month'] = (pd.to_datetime(month_df['month'], format='%m/%d/%Y').dt.to_period('m').dt.to_timestamp())
                    month_df = month_df.drop_duplicates('month').set_index('month')
                    #for duplicated last row of data
                    month_df.loc[month_df.index[-1] + pd.offsets.MonthEnd(1)] = month_df.iloc[-1]
                    month_df = month_df.resample('d').ffill()
                    ele_df = pd.concat([ele_df, month_df.reset_index()['cluster']], axis = 1)
                if set_num == 5:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\5km_ele_all.csv', index_col=0)
                if set_num == 6:
                    ele_df = pd.read_csv(r'D:\JunShen\dataset\dataAfterProcess\somFeature'+'\\'+ well_name[3:] +'\\'+'feature csv\som_feature_25cluster_SMA.csv', index_col=0)
                
                df = pd.concat([dq_well_df, ele_df], axis = 1)

                # del >30 nan
                for i in df.columns:
                    if df.loc[:,i].isnull().sum()>30:
                        n = df.loc[:,i].isnull().sum()
                        del df[i]
                        print('del', i, "nan:", n)

                main(df, model_type, well_name, set_num)