
class LocalConfig:
    urlDMI = 'https://dmigw.govcloud.dk/metObs/v1/observation'
    apiKeyPathDMI = 'src/files/apikey.txt'

    dataPath = 'data.csv'
    modelPath = 'src/modelsFiles/'

    removeFeatures = set(['index', 'datetime', 'meter', 'yearday', 'temp','dayOfYear'])
    shiftRange = [12, 48]
    shiftFeatures = ['meter']
    categoryFeatures=['dayOfYear','hourOfDay','dayOfWeek','holiday','weekend']
