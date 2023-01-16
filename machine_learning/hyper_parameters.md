# Hyper-parameters: RandomSeachCV and GridSearchCV

Hyper-parameter is a parameter whose value is used to control the learning process. Hyperparameters are the parameters that can be changed in the model to get the best-suited values.

Hyper-parameter tuning is choosing a set of optimal hyper-parameter for learning algorithm. Hyper-parameter tuning is also called hyper-parameter optimization.

Different types of hyper-parameters techniques in different models:

- Maximum depth of decision tree

- No of trees in random forest

- K in K-nearest neighbor

- Learning rate in gradient descent

- C and sigma in support vector machines

- The penalty in logistic regression classifier i.e. L1 or L2 regularization

- The Learning rate for training a neural network

## GridSearchCV

GridSearchCV searches for the best set of hyper-parameter from a grid of hyper-parameter values. The cross-validation method is used to find the training and testing set for the target estimator(model).

## RandomizedSearchCV

## Bayesian Optimization -Automate Hyper Parameter Tuning (Hyper-optimization)