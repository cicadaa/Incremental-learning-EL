
class LocalConfig:
    urlDMI = 'https://dmigw.govcloud.dk/metObs/v1/observation'
    apiKeyPathDMI = 'src/files/apikey.txt'

    dataPath = 'data.csv'
    modelPath = 'src/modelsFiles/'

    removeSet = set(['index', 'datetime', 'meter'])

    features = ['meter']
    shiftRange = [12, 36]
