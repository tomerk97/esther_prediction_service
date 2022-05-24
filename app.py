#!/usr/bin/env python
# coding: utf-8

# ### Esther project model
# 
# - Prevention of violence against women by pre-estimation of the situation from replies of a woman to questions.
# 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[2]:


# Load data
data = pd.read_csv('esther_data.csv')


# In[3]:


# Translate columns from Russian :-)))
print(data.info())
print(data.describe())
data.head()


# In[4]:


#data['target'] = data['תוצאה'] / max(data['תוצאה'])
data['target'] = data['תוצאה']
data.drop('תוצאה', axis=1, inplace=True)
data.head()


# In[5]:


data.head()


# ### Let's train some models now that we have the dataset

# In[6]:


data = data.dropna().reset_index(drop=True)


# In[ ]:





# In[7]:


target_column = 'target'
target = data[target_column]
features = data.drop(target_column, axis=1)


# In[8]:


# Split the data into train/validate and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=12345)

features_train_no_valid = features_train.copy()
target_train_no_valid = target_train.copy()

features_train, features_valid, target_train, target_valid = train_test_split(
    features_train, target_train, test_size=0.25, random_state=12345)


# In[9]:


print("Testing multiple models...\n\n")


# ### Model 1: Linear Classifier

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

clf = Perceptron(random_state=12334)
clf.fit(features_train, target_train)
y_pred = clf.predict(features_test)

print("Accuracy: ", accuracy_score(target_test, y_pred))


# ### Model 2: Random Forest Classifier

# In[11]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X = features_train
y = target_train
# define dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)

# define the model
model = RandomForestClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Training data Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

n_scores = cross_val_score(model, features_valid, target_valid, scoring='accuracy', cv=cv, n_jobs=-1,
                           error_score='raise')
# report performance
print('Validation data Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

n_scores = cross_val_score(model, features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1,
                           error_score='raise')
# report performance
print('Test data Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
#row = [[-8.52381793,5.24451077,-12.14967704,-2.92949242,0.99314133,0.67326595,-0.38657932,1.27955683,-0.60712621,3.20807316,0.60504151,-1.38706415,8.92444588,-7.43027595,-2.33653219,1.10358169,0.21547782,1.05057966,0.6975331,0.26076035]]
#yhat = model.predict(row)

yhat = model.predict(features_test)
print('Predicted Class (test data): ', yhat)


# In[26]:


# Example: How to make a single prediction?
#print("Test data: ", features_test)

test_data_index = 10
row = [features_test.loc[test_data_index]]
yhat = model.predict(row)
print("number of test samples =", len(features_test))
print("A test sample for prediction should look like this:", row)

print("\nMaking prediction... \n")
print("Single prediction test: prediction={}, real_value={}".format(yhat, target_test[test_data_index]))


model_features_sample = features_test.loc[test_data_index].copy()

#### Flask web server

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)


quarks = [{'name': 'up', 'charge': '+2/3'},
          {'name': 'down', 'charge': '-1/3'},
          {'name': 'charm', 'charge': '+2/3'},
          {'name': 'strange', 'charge': '-1/3'}]

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message' : 'Hello, World!'})

@app.route('/predict', methods=['POST'])
def addOne():
    new_request = request.get_json()
    print("Received POST request: ", new_request)
    
    for key, value in new_request.items():
        #new_request[key] = float(value)
        model_features_sample[key] = float(value)

    
    row = [model_features_sample]
    yhat = model.predict(row)

    print("A test sample for prediction should look like this:", row)

    print("\nMaking prediction... \n")
    print("Single prediction test: prediction={}".format(yhat))

    return jsonify({'prediction <class>' : str(yhat)})


if __name__ == "__main__":
    print("Starting flask server...")
    app.run(debug=False)
    print("Flask server exited...")
