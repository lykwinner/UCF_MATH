import csv
import torch
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch import nn
import math
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dates = pd.date_range('2020-1-2', '2020-12-31', freq='B')
df1 = pd.DataFrame(index=dates)
df_exxon = pd.read_csv("data/ExxonWholeYear.csv", parse_dates=True, index_col=0)
df_exxon = df1.join(df_exxon)

df_news = pd.read_csv('data/news.csv', parse_dates=True, index_col=0)
df_news = df1.join(df_news)
df_news = df_news.fillna(0)
# print(df_news)

df_exxon = df_exxon[['Exxon_stock_price']]
df_exxon.info()
df_exxon = df_exxon.fillna(method='ffill')

# print(df_exxon)

df_exxon = df_exxon.join(df_news['senti_score'])
print(df_exxon)

# df_exxon[['Exxon_stock_price']].plot(figsize=(10, 6))
# plt.ylabel("stock_price")
# plt.title("Exxon Stock Price")
# plt.show()


scaler = MinMaxScaler(feature_range=(-1, 1))
df_exxon['senti_score'] = scaler.fit_transform(df_exxon['senti_score'].values.reshape(-1, 1))
df_exxon['Exxon_stock_price'] = scaler.fit_transform(df_exxon['Exxon_stock_price'].values.reshape(-1, 1))


# function to create train, test data given stock data and sequence length
def load_data(stock, look_back, testRate):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(testRate * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]

testRate = 0.5
look_back = 30  # choose sequence length
x_train, y_train, x_test, y_test = load_data(df_exxon, look_back, testRate)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

print(y_test)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(), x_train.size())

n_steps = look_back-1
batch_size = 16
num_epochs = 100 #n_iters / (len(train_X) / batch_size)

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

# Build model
#####################
input_dim = 2
hidden_dim = 32
num_layers = 1
output_dim = 2


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
#####################

hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim = look_back - 1

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()

    # Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

# plt.plot(hist, label="Training loss")
# plt.legend()
# plt.show()

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())



# y_test_pred = scaler.inverse_transform([ytestpred[:,0]])
# y_test = scaler.inverse_transform([ytest[:,0]])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# print(np.square(np.subtract(y_test[:,0],y_test_pred[:,0])))

# print(y_test)

# Visualising the results
# figure, (axes1, axes2) = plt.subplots(2,1,figsize=(15, 6))
figure, axes1 = plt.subplots(figsize=(8, 4.5))
axes1.xaxis_date()
figure.suptitle(f'Exxon Stock Price Prediction, Test Rate: {testRate:>0.1f}, Window Size: {look_back:>1d}')


axes1.plot(df_exxon[len(df_exxon)-len(y_test):].index, y_test[:,0], color = 'orange', label = 'Real Exxon Stock Price')
axes1.plot(df_exxon[look_back:len(y_train)+look_back].index, y_train[:,0], color = 'orange')
axes1.plot(df_exxon[len(df_exxon)-len(y_test):].index, y_test_pred[:,0], color = 'red', label = 'Predicted Exxon Stock Price (Test)')
axes1.plot(df_exxon[look_back:len(y_train)+look_back].index, y_train_pred[:,0], color = 'blue', label = 'Predicted Exxon Stock Price (Train)')


axes1.set_xlabel('Time')
axes1.set_ylabel('Exxon Stock Price')
axes1.legend()

# axes2.plot(df_exxon[len(df_exxon)-len(y_test):].index, np.square(np.subtract(y_test[:,0],y_test_pred[:,0])), label = 'Square Error')
# axes2.legend()
plt.savefig(f'ExxonNewsWindowSize{look_back}.png')
plt.show()

#
# plt.close()
#
#
# # Visualising the results
# figure, (axes1, axes2) = plt.subplots(2,1,figsize=(15, 6))
# axes1.xaxis_date()
# figure.suptitle(f'Exxon Stock Price Prediction, Test Rate: {testRate:>0.1f}')
#
#
# axes1.plot(df_exxon[:len(y_train)].index, y_train[:,0], color = 'red', label = 'Real Exxon Stock Price')
# axes1.plot(df_exxon[:len(y_train)].index, y_train_pred[:,0], color = 'blue', label = 'Predicted Exxon Stock Price')
# axes1.set_xlabel('Time')
# axes1.set_ylabel('Exxon Stock Price')
# axes1.legend()
#
# axes2.plot(df_exxon[:len(y_train)].index, np.square(np.subtract(y_train[:,0],y_train_pred[:,0])), label = 'Square Error')
# axes2.legend()
# plt.savefig('StockPriceNewsTrain.png')