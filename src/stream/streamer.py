from .dataRetriever import DMIRetriever, SQLRetriever
import pandas as pd
from dateutil import tz
from datetime import datetime

__all__ = ['Streamer']


class Streamer:
    def __init__(self, path, url):
        self.dmiRetriever = DMIRetriever(path=path, url=url)
        self.sqlRetriever = SQLRetriever()

    def getChunck(self, startDate, endDate) -> pd.DataFrame:
        # 'temp_mean_past1h', 'temp_dry'-> every 10 min

        # clean temperature data frame
        dfTemp = self.dmiRetriever.getWeatherData(
            startDate=startDate, endDate=endDate, stationId="06123", field='temp_mean_past1h', limit='100000')
        dfTemp.columns = ['temp', 'datetime']
        dfTemp = dfTemp.set_index('datetime')
        dfTemp.sort_index(ascending=True)

        # clean consumptiond dataframe
        dfConsumption = self.sqlRetriever.getConsumption(columns=['datetime', 'sum'], table='consumptionAggregated',
                                                         startDate=startDate, endDate=endDate)
        dfConsumption.columns = ['datetime', 'meter']
        dfConsumption['datetime'] = dfConsumption['datetime'].apply(
            lambda x: datetime.strptime(x.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
        dfConsumption = dfConsumption.set_index(
            'datetime').resample('1H').sum()

        # merge temp and consumption data
        df = dfConsumption.join(dfTemp)

        return df
