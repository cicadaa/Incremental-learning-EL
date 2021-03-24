import pandas as pd
import requests
import json
import psycopg2
from datetime import datetime
from dateutil import tz


__all__ = ['DMIRetriever', 'SQLRetriever', 'DataRetriever']


class DMIRetriever:
    def __init__(self, path, url):
        self.keyDMI = self.initKeyDMI(path)
        self.urlDMI = url

    def initKeyDMI(self, filePath):
        with open(filePath, 'r') as f:
            key = f.read()
        return key

    def getWeatherData(self, startDate, endDate, stationId="06123", field=None, limit='100000000') -> pd.DataFrame:
        """
        get weather data from DMI
        """
        query = self.__generateDMIQuery(
            field=field, startDate=startDate, endDate=endDate, stationId=stationId, limit=limit)
        try:
            r = requests.get(self.urlDMI, params=query)
            print(r)

        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        if r.status_code != 200:
            print(r.status_code)
            raise ValueError(str(r))
        json = r.json()

        # json to dataframe
        df = pd.DataFrame(json)

        # clean data
        df['time'] = pd.to_datetime(df['timeObserved'], unit='us', utc=False)
        df = df.drop(['_id', 'timeCreated', 'timeObserved',
                      'stationId', 'parameterId'], axis=1)
        df.columns = ['temp', 'datetime']
        df = df.set_index('datetime')
        df.sort_index(ascending=True)
        return df

    def __generateDMIQuery(self, startDate, endDate, stationId, field, limit) -> dict:
        """
        Generate a dmi query
        """
        # reformat datetime
        # startDate = datetime.strptime(startDate+' +0100', '%Y-%m-%d %z')
        startDate = datetime.strptime(startDate, '%Y-%m-%d')

        endDate = datetime.strptime(endDate, '%Y-%m-%d')

        startDate = str(int(pd.to_datetime(startDate).value * 10**-3))
        endDate = str(int(pd.to_datetime(endDate).value * 10**-3))

        # create a dict for query
        query = {
            'api-key': self.keyDMI,
            'from': startDate,
            'to': endDate,
            'limit': limit
        }
        if field:
            query['parameterId'] = field
        if stationId:
            query['stationId'] = stationId
        return query


class SQLRetriever:

    def getConsumption(self, table, startDate, endDate, columns):
        try:
            connection = psycopg2.connect(user="postgres",
                                          password="read",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="postgres")

            cursor = connection.cursor()
            query = "select {0} from {1} where datetime >= '{2}' and datetime <= '{3}' order by datetime".format(
                ', '.join(columns), table, startDate, endDate)

            # fetch data
            cursor.execute(query)
            records = cursor.fetchall()

            # store data in df
            df = pd.DataFrame(index=range(len(records)), columns=columns)
            i = 0
            for index, row in df.iterrows():
                data = records[i]
                for j in range(len(columns)):
                    row[columns[j]] = data[j]
                i += 1

            # clean df
            df.columns = ['datetime', 'meter']
            df['datetime'] = df['datetime'].apply(
                lambda x: datetime.strptime(x.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
            df = df.set_index('datetime')
            return df

        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)

        finally:
            # closing database connection.
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")


class DataRetriever:
    def __init__(self, path, url):
        self.dmiRetriever = DMIRetriever(path=path, url=url)
        self.sqlRetriever = SQLRetriever()

    def getChunckData(self, startDate, endDate) -> pd.DataFrame:
        # 'temp_mean_past1h', 'temp_dry'-> every 10 min

        # clean temperature data frame
        dfTemp = self.dmiRetriever.getWeatherData(
            startDate=startDate, endDate=endDate, stationId="06123", field='temp_mean_past1h', limit='100000')

        # clean consumptiond dataframe
        dfConsumption = self.sqlRetriever.getConsumption(columns=['datetime', 'sum'], table='consumptionAggregated',
                                                         startDate=startDate, endDate=endDate)

        dfConsumption = dfConsumption.resample('1H').sum()

        # merge temp and consumption data
        df = dfConsumption.join(dfTemp)

        return df
