from pymongo import MongoClient
import pandas as pd 
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt

engine = create_engine('')


# df = pd.read_sql_table('current', engine)
df = pd.read_csv('Hotel_Reviews.csv')
# df = pd.read_csv('Hotel_Reviews.csv')

df

# connect to MongoDB, 
client = MongoClient("localhost:27017")
db=client.indv

client.list_database_names()

db.reviews.insert_many(df.to_dict('records'))

