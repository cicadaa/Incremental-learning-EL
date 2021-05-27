import torch
import torch.nn as nn
from sklearn.svm import SVR
from torch.optim import Adam
from torch.autograd import Variable
from sklearn import preprocessing as pre
from sklearn.linear_model import SGDRegressor


class SVRModel:

    def __init__(self, modelPath, updateStatus=False, acceptable=True, kernel='rbf', C=10, gamma=0.04, epsilon=.01):
        self.modelPath = modelPath
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = pre.StandardScaler()

    def learn(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class OSVRModel:

    def __init__(self, learning_rate='constant', eta0=0.4, loss='epsilon_insensitive', penalty='l2'):
        self.model = SGDRegressor(
            learning_rate=learning_rate, eta0=eta0, loss=loss, penalty=penalty)

    def learn(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class OLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(OLSTM, self).__init__() 
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(x.size())
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


class DeepTrainer:
    def __init__(self, learningRate, model):
        self.model = model
        self.lr = learningRate
        self.criterion = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = Adam(self.model.parameters(), lr=learningRate)

    def learn(self, X, y):
        outputs = self.model(X)
        self.optimizer.zero_grad()
 
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
