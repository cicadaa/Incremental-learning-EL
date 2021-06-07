# from distutils import version
from torch.nn.modules.module import T
from .config import LocalConfig
from .runner import Runner
from .models import OSVRModel, OLSTM
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = LocalConfig.dataPath
    categoryFeatures = LocalConfig.categoryFeatures
    shiftFeatures = LocalConfig.shiftFeatures
    shiftRange = LocalConfig.shiftRange
    removeFeatures = LocalConfig.removeFeatures
    learning_rate = 0.01 #best rate
    
    # osvr = OSVRModel(learning_rate='constant', eta0=0.4,
    #                   loss='epsilon_insensitive', penalty='l2')

    # olstm = OLSTM(num_classes=1, input_size=1, hidden_size=24, num_layers=1)

    # dataset_osvr = Dataset(dataPath=dataPath, shiftFeatures=['meter'], 
                            # categoryFeatures=['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday'],
                            # shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False, useTimeFeature=False)

    # dataset_olstm = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=categoryFeatures,
    #                   shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=True, useTimeFeature=False)
    # dataset_osvr.getTrainData(0,1)
    # dataset.getTrainData(1,2)

    # #osvr
    # runner_osvr = Runner(warmStartPoint=1, dataset=dataset_osvr, model=osvr, deep=False, learningRate=learning_rate, lazy=False)
    # runner.run(duration=30, interval=0, name='osvr', plot=True, record=True, verbose=True)
    # '''best-0.0425'''


    # #osvr
    # runner_osvr = Runner(warmStartPoint=1, dataset=dataset_osvr, model=osvr, deep=False, learningRate=learning_rate, lazy=False)
    # runner.run(duration=30, interval=0, name='osvr', plot=True, record=True, verbose=True)
    # '''best-0.0425'''


    #TESTS===========================================================================================================

    #01-olstm ************************************************************************************
    '''
    best result: 
    01_olstm mape: 
    0.07626588975022867
    01_olstm runnint time: 
    646.5130829811096

    lazy: False
    features: lag feature [12, 36] + ['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday']
    learn one: learn 48 batches
    warm start: 1
    learning rate:0.01
    hidden size: 168
    input size: 5

    '''
    # model = OLSTM(num_classes=1, input_size=5, hidden_size=168, num_layers=1)
    # dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=categoryFeatures,
    #                   shiftRange=[12,36], removeFeatures=removeFeatures, isTorch=True, useTimeFeature=True)

    # runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=True, learningRate=0.01, lazy=False)
    # runner.run(duration=1000, interval=0, name='01_olstm', plot=True, record=True, verbose=True)


    #02-olstm - time series only  ************************************************************************************
    '''
    best result: 
    02_olstm mape: 
    0.04932617872727324
    02_olstm runnint time: 
    149.98655915260315

    lazy: False
    features: lag feature [12, 36] 
    learn one: learn 48 batches
    warm start: 1
    learning rate:0.01
    hidden size: 24
    input size: 1
    '''
    # model = OLSTM(num_classes=1, input_size=1, hidden_size=24, num_layers=1)
    # dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=categoryFeatures,
    #                   shiftRange=[12,36], removeFeatures=removeFeatures, isTorch=True, useTimeFeature=False)

    # runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=True, learningRate=0.01, lazy=False)
    # runner.run(duration=1000, interval=0, name='02_olstm', plot=True, record=True, verbose=True)


    #03-lolstm - lazy approach  ************************************************************************************
    '''
    best result: 
    
    03_lolstm mape: 
    0.0893665755172856
    03_lolstm runnint time: 
    231.88226413726807
    training times: 
    37.83457782883401 %

    lazy: True
    features: lag feature [12, 36] + ['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday']
    learn one: learn 48 batches
    learn many/lazy num: 48
    lazyThreshold=0.07
    warm start: 1
    learning rate:0.01
    hidden size: 168
    input size: 5
    '''
    model = OLSTM(num_classes=1, input_size=5, hidden_size=168, num_layers=1)
    dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=categoryFeatures,
                      shiftRange=[12,36], removeFeatures=removeFeatures, isTorch=True, useTimeFeature=True)

    runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=True, learningRate=0.01, lazy=True)
    runner.run(duration=2000, interval=0, lazyThreshold=0.07, lazyNum=48, name='03_lolstm', plot=True, record=True, verbose=True)

    
    #04-osvr  ************************************************************************************
    '''
    best result: 
    04_osvr mape: 
    0.06562833695865494
    04_osvr runnint time: 
    34.02925086021423
    lazy: False
    features: lag feature [12, 36] + ['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday']
    learn one: learn one row
    warm start: 1
    eta0:0.9
    '''
    # model = OSVRModel(learning_rate='constant', eta0=0.9, loss='epsilon_insensitive', penalty='l2')
    # dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=['dayOfYear','hourOfDay', 'dayOfWeek', 'holiday'],
    #                   shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False)

    # runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=False, learningRate=0.01, lazy=False)
    # runner.run(duration=1000, interval=0, name='04_osvr', plot=True, record=True, verbose=True)

    #05-osvr - time series only  ************************************************************************************
    '''
    best result: 
    05_osvr mape: 
    0.04287276347600463
    05_osvr runnint time: 
    33.80311703681946
    lazy: False
    features: lag feature [12, 36] 
    learn one: learn one row
    warm start: 1
    eta0:0.4
    '''
    # model = OSVRModel(learning_rate='constant', eta0=0.4, loss='epsilon_insensitive', penalty='l2')
    # dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=[],
    #                   shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False, useTimeFeature=False)

    # runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=False, learningRate=0.01, lazy=False)
    # runner.run(duration=1000, interval=0, name='05_osvr', plot=True, record=True, verbose=True)


    #06-losvr - lazy approach ************************************************************************************
    '''
    best result: 
    06_osvr mape: 
    0.06407375937736526
    06_osvr runnint time: 
    113.38295483589172
    training times: 
    29.982768523836878 %
    lazy: True
    features: lag feature [12, 36] 
    learn one: learn one row
    learn many/lazy num: 96
    lazyThreshold: 0.08
    warm start: 1
    eta0:0.9
    '''
    # model = OSVRModel(learning_rate='constant', eta0=0.9, loss='epsilon_insensitive', penalty='l2')
    # dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=[],
    #                   shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False, useTimeFeature=False)

    # runner = Runner(warmStartPoint=1, dataset=dataset, model=model, deep=False, learningRate=0.01, lazy=True)
    # runner.run(duration=1000, interval=0, lazyThreshold=0.07, lazyNum=48, name='06_osvr', plot=True, record=True, verbose=True)