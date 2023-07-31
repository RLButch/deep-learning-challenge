# deep-learning-challenge

Overview
This deep learning challenge aimed to develop a binary classifier that can predict the likelihood of applicants achieving success if they receive funding from the charity - Alphabet Soup. The project utilized features present in the given dataset and employ diverse machine learning methods to train and assess the model's performance. The objective of this challenge was to optimize the model in order to attain an accuracy score of 75% or more.

Results


Data Preprocessing
The model aims to predict the success of applicants if they receive funding from Alphabet soup. This is indicated by the IS_SUCCESSFUL column in the dataset which is the target variable of the model. The feature variables are every column other than the target variable and the non-relevant variables such as EIN and names. The features capture relevant information about the data and can be used in predicting the target variables, the non-relevant variables that are neither targets nor features will be drop from the dataset to avoid potential noise that might confuse the model.
During preprocessing, I implemented binning/bucketing for rare occurrences in the APPLICATION_TYPE and CLASSIFICATION columns. Subsequently, the categorical data was transformed into numeric data using the one-hot encoding technique. Then the data was split into separate sets for features and targets, as well as for training and testing. Lastly, the data was scaled in order to ensure uniformity in the data distribution.


Initial Model optimisation: For the initial model optimisation, 3 layers were included: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron. This choice was made in order to ensure that the total number of neurons in the model was between 2-3 times the number of input features. In this case, there were 43 input features remaining after removing 2 irrelevant ones. After this the relu activation function was selected for the first and second layers, and the sigmoid activation function for the output layer since the goal was binary classification. To start, themodel was trained for 100 epochs and achieved an accuracy score of approximately 72.9. I don't believe that there was any apparent indication of overfitting or underfitting.

Then for the next optimisation -  optimized the model’s performance by first modified the architecture of the model by adding 2 dropout layers with a rate of 0.5 to enhance generalization and changed the activation function to tanh in the input and hidden layer. With that I got an accuracy score of 73%.
I used hyperparameter tuning. During this process, Keras identified the optimal hyperparameters, which include using the tanh activation function, setting 41 neurons for the first layer, and assigning 13, 5, and 4 units for the subsequent layers. As a result, the model achieved an accuracy score of 73%.

Optimisation 3
Method 3 – 72% Accuracy, 64% Loss
🔸 Two hidden relu layers
🔸 100 epochs to train the model

Summary

In summary, the target accuracy value of 75% was not achieved, however our first and second optimization got close to it at 73%. To further improve the accuracy, more EDA is potentially required for the input and target data. Since I couldnt' achieve the accuracy of 75% I'd suggest using alternative models. Perhaps  making changes to the dropout layers and trying different activation functions, while adjusting layers and neurons could improve the model optimisation to achieve an accuracy of 75%. 


