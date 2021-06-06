import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


__all__ = ['loadModel', 'plotResult', 'plotlyplot']

# Model Management=============================================================


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# Visualization ===============================================================


def plotResult(actual, prediction, times, figsize=(26, 10)):
    plt.style.use('seaborn')
    month_day_fmt = mdates.DateFormatter('%b %d')

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(times, actual, label='Actual', color='blue')
    ax.plot(times, prediction, label='Prediction', color='red')
    ax.legend()
    ax.xaxis.set_major_formatter(month_day_fmt)
    plt.show()


def plotlyplot(actual, prediction, times, plotname):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=actual,
                             mode='lines',

                             name='actual'))
    fig.add_trace(go.Scatter(x=times, y=prediction,
                             mode='lines',
                             name='prediction'))

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
        )
    )
    fig.show()
    fig.write_html("/Users/cicada/Documents/DTU_resource/Thesis/Incremental-learning-EL/src/results/results-"+plotname+".html")


# Time Formater ===============================================================

def getNextTime(start, interval):
    timeFormat = "%Y-%m-%d %H:%M:%S"
    end = datetime.strptime(start, timeFormat) + timedelta(hours=interval)
    return end.strftime(timeFormat)
