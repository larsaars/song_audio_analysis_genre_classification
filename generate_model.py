"""
create the model from database
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import global_variables as gv

# read df, make genres to str lists and drop all empty genres lists then
df = pd.read_csv(gv.db_file, sep=';')

# make nan to 0 in every column
for column in df:
    if column != 'genre':
        df[column] = df[column].fillna(0)


# print the header of the df
print(df.head())

# create text train split
train, test = train_test_split(df, test_size=0.2)
# create test and train x, y
# X is data, y is target
trainY, testY = train.genre, test.genre
trainX, testX = train.drop(columns=['genre']), test.drop(columns=['genre'])

# create the model
model = RandomForestClassifier(n_estimators=150)  # verbose = 2
# fit linear regression
model.fit(X=trainX, y=trainY)

# dump model into file after generating
with open(gv.model_file, 'wb') as file:
    pkl.dump(model, file)

# print test score
print('score: %f' % model.score(X=testX, y=testY))
