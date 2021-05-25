import time
from .utils import *
import pandas as pd
from .config import LocalConfig
from .models import DeepTrainer
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, warmStartPoint, dataset, model, deep=False, learningRate=0.01):
        self.cur = None
        self.nxt = None
        
        self.times = dataset.times
        self.yTrue = dataset.y
        self.dataset = dataset

        self.model = model
        self.startPont = warmStartPoint
        self.predList, self.actualList, self.scoreList = [], [], []
        self.isdeep = deep
        if self.isdeep:
            self.trainer = DeepTrainer(learningRate=learningRate, model=self.model)

    def _warmStart(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=0, idxTo=self.startPont)

        if self.isdeep:
            self.trainer.learn(XTrain, yTrain)
        else:
            self.model.learn(XTrain, yTrain)

    def _evaluate(self, method, range, baseScore):
        idxFrom, idxTo = self.cur - range, self.cur
        if self.model.acceptable:
            yPred, yActual = self.predList[idxFrom:
                                           idxTo], self.actualList[idxFrom: idxTo]
            if method == 'r2':
                score = r2_score(y_pred=yPred, y_true=yActual)
                self.model.acceptable = False if score < baseScore else True
            elif method == 'mape':
                score = mean_absolute_percentage_error(
                    y_pred=yPred, y_true=yActual)
                self.model.acceptable = False if score > baseScore else True
            self.scoreList.append(score)

    def _predict(self,log=True):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        
        if self.isdeep:
            yPred = self.model.forward(XTrain).data.numpy()
            yPred = self.dataset.scaler.inverse_transform(yPred)
            yTrue = self.yTrue[self.cur]
        else:
            yPred = self.model.predict(XTrain)

        if log:

            self.predList.append(yPred[0][0])
            self.actualList.append(yTrue)

    def _learn(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        if self.isdeep:
            self.trainer.learn(XTrain, yTrain)
        else:
            self.model.learn(XTrain, yTrain)

    def _update(self):
        self.cur += 1
        self.nxt = self.cur + 1

    def run(self, duration, name, interval, evaluate=False, deep=False, plot=False, record=True):

        begin = time.time()
        self._warmStart()
        self.cur = self.startPont

        # streaming
        while time.time() - begin < duration:
            time.sleep(interval)
            self._update()
            self._predict()

            # evaluate model
            acceptable = self._evaluate('r2', 24, 0.7) if evaluate else False
            if not acceptable:
                self._learn()

        if plot:
            plotlyplot(actual=self.actualList, prediction=self.predList,
                   times=self.times[:self.cur-1], plotname=name)

        if record:
            # actualvalue = self.actualList
            # predictvalue = self.predList
            # timesidx = self.times[:self.cur-1]
            
            # dictionary of lists 
            dict = {'actual': self.actualList, 'predict': self.predList, 'time': self.times[:self.cur-1]}         
            df = pd.DataFrame(dict)
            df.to_csv(name + 'result.csv')
