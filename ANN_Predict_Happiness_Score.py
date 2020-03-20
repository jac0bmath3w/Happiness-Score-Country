#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:03:52 2020
Predicting Happiness Score based on 7 unique variables
@author: jacob
"""


# Import the dataset as a Pandas dataframe
import pandas as pd
import matplotlib.pyplot as plt
# Hyperparameter tuning

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#For feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScalar

#For MSE
from sklearn.metrics import mean_squared_error

data = pd.read_csv('World_Happiness_2015_2017.csv')


# Decide the variables to make the prediction on. 
# The variables Country and Year are not included in this analysis.
# Therefore, this is not a temporal or a spatial analysis
# The variable Happiness rank is rank of country based on happiness score. 
# Therefore, it is not included. The other variables are included


# Check for missing values in the data
X = data.iloc[:,3:-1]
X.columns[X.isnull().any()].tolist()


# No missing values have been identified. 

X = X.values
y = data.iloc[:,2].values

# There are no categorical data in this dataset. 
# Do the train_test_ split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature Scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)   #because already fit to the training data. X_train and X_test are fit to the same data


# Describing the NN architecture

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN

happiness_model = Sequential()

# Adding the input layer and hidden layers


happiness_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
happiness_model.add(Dropout(0.1))
happiness_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
happiness_model.add(Dropout(0.1))
happiness_model.add(Dense(units = 1))


# Compiling the ANN

happiness_model.compile(optimizer='adam', loss='mean_squared_error')

happiness_model.fit(X_train, y_train, epochs = 100, batch_size =32 )


predicted_happiness_scores= happiness_model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real Happiness Scores')
plt.plot(predicted_happiness_scores, color = 'blue', label = 'Predicted Happiness Scores')
plt.title('Happiness Score Prediction')
plt.xlabel('Index')
plt.ylabel('Happiness Score')
plt.legend()
plt.show()


#Hyperparameter tuning starts here



def build_regressor(optimizer='adam'):
    reg_model = Sequential()
    reg_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
    reg_model.add(Dropout(0.1))
    reg_model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    reg_model.add(Dropout(0.1))
    reg_model.add(Dense(units = 1))
    reg_model.compile(optimizer=optimizer, loss='mean_squared_error')
    return(reg_model)
    

parameters = {'batch_size':[25, 32], 
              'epochs':[100,250],
              'optimizer':['adam','rmsprop']}
estimator = KerasRegressor(build_fn=build_regressor, verbose=1)

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=parameters, 
                           n_jobs=-1, 
                           cv = 10)


tuned_regression_model = grid_search.fit(X_train, y_train)

best_parameters = tuned_regression_model.best_params_
best_mse = tuned_regression_model.best_score_

y_pred_tuned = tuned_regression_model.predict(X_test)

#predicted_happiness_scores= happiness_model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real Happiness Scores')
plt.plot(predicted_happiness_scores, color = 'blue', label = 'Predicted Happiness Scores')
plt.plot(y_pred_tuned, color = 'green', label = 'Predicted Happiness Scores Tuned Model')

plt.title('Happiness Score Prediction')
plt.xlabel('Index')
plt.ylabel('Happiness Score')
plt.legend()
plt.show()

#Scaling the dependent variable using Min Max Scalar and fitting a sigmoid function at the end. 
sc_Y = MinMaxScaler(feature_range=(0,1))
scaled_y_train = sc_Y.fit_transform(y_train.reshape(-1, 1))

happiness_model_scaled = Sequential()

# Adding the input layer and hidden layers
happiness_model_scaled.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
happiness_model_scaled.add(Dropout(0.1))
happiness_model_scaled.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
happiness_model_scaled.add(Dropout(0.1))
happiness_model_scaled.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the Model
happiness_model_scaled.compile(optimizer='adam', loss='mean_squared_error')

happiness_model_scaled.fit(X_train, scaled_y_train, epochs = 250, batch_size =25 )
predicted_happiness_scores_scaled = happiness_model_scaled.predict(X_test)
predicted_happiness_scores_scaled = sc_Y.inverse_transform(predicted_happiness_scores_scaled)

plt.plot(y_test, color = 'red', label = 'Real Happiness Scores')
plt.plot(predicted_happiness_scores, color = 'blue', label = 'Predicted Happiness Scores')
plt.plot(y_pred_tuned, color = 'green', label = 'Predicted Happiness Scores Tuned Model')
plt.plot(predicted_happiness_scores_scaled, color = 'yellow', label = 'Predicted Happiness Scores Scaled Model')

plt.title('Happiness Score Prediction')
plt.xlabel('Index')
plt.ylabel('Happiness Score')
plt.legend()
plt.show()



#Store saved model
#import pickle
#filename = 'initial_model.sav'
#filename = 'tuned_model.sav'
#pickle.dump(happiness_model, open(filename, 'wb'))
#pickle.dump(tuned_regression_model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#loaded_model.predict(X_test)

#Calculating the Mean Squared Error

mse_initial_model = mean_squared_error(y_test, predicted_happiness_scores)
mse_tuned_model = mean_squared_error(y_test, y_pred_tuned)
mse_scaled = mean_squared_error(y_test, predicted_happiness_scores_scaled)

