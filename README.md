# Predict-Happiness-Score
Building a neural network to predict the happiness score according to economic prodution, social score, etc. Dataset is obtained from https://www.kaggle.com/unsdsn/world-happiness

Three different models are built in this exercise. 

1. A neural network with 2 hidden layers. This is trained using Happiness score as the dependent variable and the rest of the variables as the independent variables. (Country and time are excluded). 

2. The above mentioned neural network tuned and the best parameters selected based on 10 fold cross validation. 

3. The dependent variable is scaled using the MinMaxNormalization. Now it lies between 0 and 1. The last layer of the neural net regressor is activated using the sigmoid function. 
