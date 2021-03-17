import pandas as pd
import requests
import json
import psycopg2
from datetime import datetime

__all__ = ['DMIRetriever', 'SQLRetriever']


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
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        if r.status_code != 200:
            raise ValueError(str(r))
        json = r.json()

        # json to dataframe
        df = pd.DataFrame(json)
        df['time'] = pd.to_datetime(df['timeObserved'], unit='us', utc=False)
        df = df.drop(['_id', 'timeCreated', 'timeObserved',
                      'stationId', 'parameterId'], axis=1)
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

            return df

        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)

        finally:
            # closing database connection.
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")


# if __name__ == '__main__':
#     httpRetiever = DMIRetriever(path='apikey.txt', url='')
#     df = httpRetiever.getWeatherData(startDate="2019-01-01", endDate="2019-01-02",
#                                      stationId="06123", field="temp_mean_past1h", limit='10')
#     print(df.head())

#     # sqlRetriever = SQLRetriever()
#     # df = sqlRetriever.getConsumption(columns=['datetime', 'sum'], table='consumptionAggregated',
#     #                                  startDate='2019-03-01', endDate='2019-03-02')
#     # print(df.head())
