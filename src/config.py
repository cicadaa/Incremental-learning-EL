
class LocalConfig:
    urlDMI = 'https://dmigw.govcloud.dk/metObs/v1/observation'
    apiKeyPathDMI = 'src/files/apikey.txt'

    datapath = 'data.csv'
    features = ['meter', 'temp']

    prevFrom = 12
    prevTo = 24

    modelPath = 'src/models/svrLatest.pkl'
