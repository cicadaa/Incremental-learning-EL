import psycopg2

def getHistory(table, startDate, endDate, columns):
    try:
        connection = psycopg2.connect(user="chanyuyang",
                                    password="read",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="postgres")

        cursor = connection.cursor()
        query = "select {0} from {1} where hourutc >= '{2}' and hourutc <= '{3}' limit 5".format(', '.join(columns), table, startDate, endDate)

        #fetch data
        cursor.execute(query)
        records = cursor.fetchall()
        print(records)
        return records

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    getHistory(columns=['hourutc','fossilgas'],table='test01', startDate='2018-03-01', endDate='2019-03-01')