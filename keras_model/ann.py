# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype 
import datetime
# Importing the dataset
dataset = pd.read_csv('data/train.csv')
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
dataset['clinic'] = dataset['clinic'].astype(str)

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
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X = X[:500000]
y = y[:500000]
print("Finished Data processing")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print("starting X_text")
X_test = sc.transform(X_test)



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 17))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 15)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.95)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#saving model
#serialize model to JSON
model_json = classifier.to_json()
with open("models/feed_forward/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("models/feed_forward/model.h5")
print("Saved model to disk")
