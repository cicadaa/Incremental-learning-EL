import time
from .utils import *
import pandas as pd
from .models import DeepTrainer
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, warmStartPoint, dataset, model, deep=False, learningRate=0.01, lazy=False):
        #stream indicator
        self.cur = None
        self.nxt = None    
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
        self.startPont = warmStartPoint
        self.predList, self.actualList, self.scoreList = [], [], []


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
            yPred = self.model.forward(XTrain).data.numpy()
            yPred = self.dataset.scaler.inverse_transform(yPred)[0]
            yTrain = self.yTrue[self.cur]
        else:
            yPred = self.model.predict(XTrain)
            print('ypred',yPred)

        if log:
            self.predList.append(yPred[0])
            self.actualList.append(yTrain)


    def _learnOne(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        if self.isdeep:
            self.trainer.learn(XTrain, yTrain)
        else:
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


    def run(self, duration, name, interval, plot=False, record=True, verbose=True):
        begin = time.time()
        self._warmStart()
        self.cur = self.startPont
        acceptable = False

        # streaming
        while time.time() - begin < duration:
            time.sleep(interval)
            self._update()
            self._predict()

            # evaluate model
            if self.lazy and self.cur > 12:
                acceptable = self._evaluate('mape', 12, 0.1)
                if not acceptable:
                    print('train')
                    self._learnMany(numberOfData=12)
            else:
                self._learnOne()       

        if plot:
            plotlyplot(actual=self.actualList, prediction=self.predList,
                   times=self.times[:self.cur-1], plotname=name)

        if record:
            dict = {'actual': self.actualList, 'predict': self.predList, 'time': self.times[:self.cur-1]}         
            df = pd.DataFrame(dict)
            df.to_csv(name + 'result.csv')

        if verbose:
            print(mean_absolute_percentage_error(self.actualList[200:], self.predList[200:]))
