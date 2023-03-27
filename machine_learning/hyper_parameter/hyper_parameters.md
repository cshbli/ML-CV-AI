# Hyper-parameters: Random Seach and Grid Search

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

## Issues of Grid Search and Randomized Search

Grid Search searches for the best set of hyper-parameter from a grid of hyper-parameter values. The cross-validation method is used to find the training and testing set for the target estimator(model).

- Computation power and time
- Limited and Biased Search: expanding the search space further will exponentially increase the processing time for Grid Search
- Discrete search

## Bayesian Optimization

Bayesian Optimization uses Bayesian statistics to estimate the distribution of the best hyperparameters for the model instead of just using grid search or random search.

During the tuning process, the algorithm updates its beliefs about the distribution of the best hyperparameters based on each hyperparameter's observed impact on the model's performance. This allows it to gradually converge on the optimal set of hyperparameters, resulting in better performance on the test set.

||Model Acuracy|Waiting Time|
|---|---|---|
|Grid Search|High|Very High|
|Bayesian Search On Discrete Parameters|High|Very Low|
|Bayesian Search on Continuous Parameters|Very High|Low|

## References

- [Grid search and random search are outdated. This approach outperforms both.](https://medium.com/@ali.soleymani.co/stop-using-grid-search-or-random-search-for-hyperparameter-tuning-c2468a2ff887)