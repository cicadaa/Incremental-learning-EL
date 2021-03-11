
from collections import defaultdict
import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime


class Meteorology:
    def __init__(self):
        self.key = self.__read_key()
        self.baseurl = 'https://dmigw.govcloud.dk/metObs/v1/observation'

    def __read_key(self):
        key = ''
        with open('src/stream/apikey.txt', 'r') as f:
            key = f.read()
        return key

    def get_meteodata(self, start_date, end_date, station_id="06123", field=None, limit='100000'):
        query = self.__generate_query(
            field, start_date, end_date, station_id, limit)

        try:
            r = requests.get(self.baseurl, params=query)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        if r.status_code != 200:
            raise ValueError(str(r))
        json = r.json()

        # json to dataframe
        df = pd.DataFrame(json)
        df['time'] = pd.to_datetime(df['timeObserved'], unit='us')
        df = df.drop(['_id', 'timeCreated', 'timeObserved'], axis=1)
        df.set_index = df['time']
        return df

    # get time interval in unixtime format
    def __get_timeinterval(self, start_date, end_date):
        start_date = self.__to_datetime(start_date)
        end_date = self.__to_datetime(end_date)
        return self.__datetime2unixtime(start_date), self.__datetime2unixtime(end_date)

    # convert 8-digit string to datetime
    def __to_datetime(self, date):
        if type(date) != str or len(date) != 8:
            raise TypeError
        return datetime(int(date[:4]), int(date[4:6]), int(date[6:]))

    # datetime to unixtime
    def __datetime2unixtime(self, datetime):
        return str(int(pd.to_datetime(datetime).value * 10**-3))

    # generate query parameters
    def __generate_query(self, field, start_date, end_date, station_id, limit):
        start_time, end_time = self.__get_timeinterval(start_date, end_date)
        query = {
            'api-key': self.key,
            'from': start_time,
            'to': end_time,
            'limit': limit
        }
        if field:
            query['parameterId'] = field
        if station_id:
            query['stationId'] = station_id
        return query

if __name__ == '__main__':
    dmi = Meteorology()
    df = dmi.get_meteodata("20190101","20190102","06123", "temp_mean_past1h",'10')
    print(df.head())