
class LocalConfig:
    urlDMI = 'https://dmigw.govcloud.dk/metObs/v1/observation'
    apiKeyPathDMI = 'src/files/apikey.txt'

    dataPath = '/Users/cicada/Documents/DTU_resource/Thesis/Incremental-learning-EL/src/data-withtemp.csv'

    removeFeatures = set(['index', 'temp', 'Unnamed: 0', 'Unnamed: 0.1'])
    shiftRange = [12, 36]
    shiftFeatures = ['meter']
    categoryFeatures = []
    # categoryFeatures=['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday', 'weekend']
