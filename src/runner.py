import time
from .utils import *
import pandas as pd
from .models import DeepTrainer
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, warmStartPoint, dataset, model, deep=False, learningRate=0.01, lazy=False):
        #stream indicator
        self.cur = 0
        self.nxt = 1    
        #training data 
        self.times = dataset.times
        self.yTrue = dataset.y
        self.dataset = dataset
        #model
        self.model = model
        self.isdeep = deep
        if self.isdeep:
            self.trainer = DeepTrainer(learningRate=learningRate, model=self.model)
        #runner config
        self.lazy = lazy
        self.learnManyTimes = []
        self.startPont = warmStartPoint
        self.predList, self.actualList, self.timeline, self.scoreList = [], [], [], []


    def _warmStart(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=0, idxTo=self.startPont)

        if self.isdeep:
            self.trainer.learn(XTrain, yTrain)
        else:
            self.model.learn(XTrain, yTrain)


    def _evaluate(self, method, range, baseScore):
        idxFrom, idxTo = self.cur - range, self.cur
        yPred, yActual = self.predList[idxFrom:idxTo], self.actualList[idxFrom: idxTo]
        if method == 'r2':
            score = r2_score(y_pred=yPred, y_true=yActual)
            return False if score < baseScore else True
        elif method == 'mape':
            score = mean_absolute_percentage_error(y_pred=yPred, y_true=yActual)
            return False if score > baseScore else True


    def _predict(self,log=True):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        
        if self.isdeep:
            yPred = self.model.forward(XTrain).data.numpy()[0]
            # yPred = self.dataset.deepScaler.inverse_transform(yPred)[0]
            # yTrain = yTrain[0][0]
            yTrain = self.yTrue[self.cur][0]

        else:
            yPred = self.model.predict(XTrain)
            yTrain = self.yTrue[self.cur]
            # print(yPred)

        if log:
            self.predList.append(yPred[0])
            self.actualList.append(yTrain)
            self.timeline.append(self.times[self.cur])


    def _learnOne(self):
        if self.isdeep:
            XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur-48, idxTo=self.nxt)
            self.trainer.learn(XTrain, yTrain)
        else:
            XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
            self.model.learn(XTrain, yTrain)


    def _learnMany(self, numberOfData):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.nxt-numberOfData, idxTo=self.nxt)
        if self.isdeep:
            self.trainer.learn(XTrain, yTrain)
        else:
            self.model.learn(XTrain, yTrain)


    def _update(self):
        self.cur += 1
        self.nxt = self.cur + 1


    def run(self, duration, name, interval, lazyNum=24, lazyThreshold=0.1, plot=False, record=True, verbose=True):
        begin = time.time()
        self._warmStart()
        self.cur = self.startPont
        acceptable = False

        # streaming
        while time.time() - begin < duration and self.nxt < len(self.yTrue):
            # time.sleep(interval)
            self._update()
            self._predict()
        

            # evaluate model
            if self.lazy and self.cur > lazyNum:
                acceptable = self._evaluate('mape', 12, lazyThreshold)
                if not acceptable:
                    self.learnManyTimes.append(self.times[self.cur])
                    self._learnMany(numberOfData=lazyNum)
            else:
                self._learnOne()   
        end = time.time()
        if plot:
            plotlyplot(actual=self.actualList, prediction=self.predList,
                   times=self.times[:self.cur-1], plotname=name)

        if record:         
            dict = {'actual': self.actualList, 'predict': self.predList, 'time': self.timeline}        
            df = pd.DataFrame(dict)
            df.to_csv('/Users/cicada/Documents/DTU_resource/Thesis/Incremental-learning-EL/src/results/results-'+name + '.csv')
            
        if verbose:
            print(name+ ' mape: ')
            print(mean_absolute_percentage_error(self.actualList[600:2000], self.predList[600:2000]))
            print(name+ ' runnint time: ')
            print(end-begin)
            if self.lazy:
                print('training times: ')
                print(len(self.learnManyTimes)/len(self.timeline)*100, '%')