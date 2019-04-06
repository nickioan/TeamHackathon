
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype 
import datetime
# Importing the dataset
dataset = pd.read_csv('data/test.csv')
#dataset = dataset.dropna()

#Preprocessing Appointment Timestamps
dataset['date'] = pd.to_datetime(dataset.apt_date)
dataset['apt_month'] = dataset['date'].dt.month
dataset['apt_hour'] = dataset['date'].dt.hour
dataset = dataset.drop(columns=['date', 'apt_date'])

# Preprocessing Sent Timestamps
dataset['date'] = pd.to_datetime(dataset.sent_time)
dataset['sent_month'] = dataset['date'].dt.month
dataset['sent_hour'] = dataset['date'].dt.hour
dataset = dataset.drop(columns=['date', 'sent_time'])

dataset = dataset.drop(
    columns=['cli_zip', 'pat_id', 'family_id', 'send_time', 'ReminderId'])
print(dataset.columns)

for column in dataset:
    if is_string_dtype(dataset[column]):
        dataset[column] = dataset[column].fillna(value='NaN')
    else:
        dataset[column] = dataset[column].fillna(dataset[column].mean)

all_cats = {}
for column in dataset:
    x = dataset[column].unique()
    all_cats[column] = x



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for column in dataset:
    if column in ['response', 'apt_type', 'net_hour', 'type', 'clinic', 'city',
       'province', 'cli_area', 'cli_size', 'fam', 'gender', 'age', 'pat_area',
       'dist', 'apt_month', 'apt_hour', 'sent_month', 'sent_hour']:
        lencoder = LabelEncoder()
        lencoder.fit(all_cats[column])

        dataset[column] = lencoder.transform(dataset[column])
        '''
        one = OneHotEncoder(n_values=len(all_cats[column]), sparse=False)
        p=dataset[column]
        p = p.
        p=p.reshape(len(p), 1) 
        a = one.fit_transform(dataset[column])
        dataset[column]  = a
        print(dataset[column])
        '''
X = dataset.iloc[:, 0:].values

print("Finished Data processing")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from keras.models import model_from_json

json_file = open('models/feed_forward/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/feed_forward/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
y_pred = loaded_model.predict(X)
print(str(len(y_pred)))
print(y_pred)
for i, y in enumerate(y_pred):
    if y > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

import csv
f = open("results.csv", 'w')
f.write("ReminderId, response\n")
for i in range(len(y_pred)):
    row = str(i) + "," + str(int(y_pred[i][0])) + "\n"
    f.write(row) 