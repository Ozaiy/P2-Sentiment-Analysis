from pymongo import MongoClient
from bson.code import Code
import pandas as pd 

client = MongoClient("localhost:27017")
db=client.indv


map = Code("function () {"
            "    emit(this, 1);"
            "}")


reduce = Code("function (key, values) {"
               "  var total = 0;"
               "  for (var i = 0; i < values.length; i++) {"
               "    total += values[i];"
               "  }"
               "  return total;"
               "}")

result = db.labeled.map_reduce(map, reduce, "myresults")

rmv_duplicated = []

for doc in result.find():
    if doc['value'] < 2:
        rmv_duplicated.append(doc)



df = pd.DataFrame(rmv_duplicated)

source = list(df['_id'].values)

data = pd.DataFrame(source)

df.columns = ['Review']


