from sklearn.svm import SVR


class LocalConfig:
    urlDMI = 'https://dmigw.govcloud.dk/metObs/v1/observation'
    apiKeyPathDMI = 'src/files/apikey.txt'

    model = SVR(kernel='rbf', C=10, gamma=0.04, epsilon=.01)
    datapath = 'data.csv'
    features = ['meter']

    prevFrom = 12
    prevTo = 24

    modelPath = 'src/models/svrLatest.pkl'
