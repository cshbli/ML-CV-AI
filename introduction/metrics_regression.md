# Evaluaton for Regression Models

## Mean Absolute Error (MAE)
$$ MAE = \dfrac{1}{N}\sum_{i=1}^N|y_i - \hat y_i|$$

### Advantages
- MAE is not sensitive to outliers. Use MAE when you do not want outliers to play a big role in error calculated.

### Disadvantages
- MAE is not differentiable globally. This is not convenient when we use it as a loss function, due to the gradient optimization method.

## Meas Squared Error (MSE)
$$ MSE = \dfrac{1}{N}\sum_{i=1}^N(y_i - \hat y_i)^2$$

### Advantages
- MSE is differantiable which means it can be easily used as a loss function.

- MSE can be decomposed into variance and bias squared. This helps us understand the effect of variance or bias in data to the overall error.

$$ MSE(\hat y) = Var(\hat y) + (Bias(\hat y))^2$$

### Disadvantages
- The value calculated MSE has a different unit than the target variable since it is squared. (Ex. meter → meter²)

- If there exists outliers in the data, then they are going to result in a larger error. Therefore, MSE is not robust to outliers (this can also be an advantage if you are looking to penalize outliers).

## Root Mean Squared Error (RMSE)

$$ RMSE = \sqrt{\dfrac{1}{N}\sum_{i=1}^N(y_i - \hat y_i)^2}$$

### Advantages
- The error calculated has the same unit as the target variables making the interpretation relatively easier.

### Disadvantages
- Just like MSE, RMSE is also susceptible to outliers.

## R-Squared ($R^2$)

$$ R^2 = \dfrac{Variance\quad considered\quad by\quad model}{Total\quad Variance}$$

$$ R^2 = 1 - \dfrac{SS_{regression}}{SS_{total}} = 1 - \dfrac{\sum(y_i - \hat y_i)2}{\sum(y_i - y_{mean})^2}$$

### Advantages
- R-square is a handy, and an intuitive metric of how well the model fits the data. Therefore, it is a good metric for a baseline model evaluation. 

### Disadvantages
- R-squared can’t determine if the predictions are biased, that is why looking at the residual plots in addition is a good idea.

- R-squared does not necessarily indicate that a regression model is good to go. It is also possible to have a low R-squared score for a good regression model and a high R-squared model for a bad model (especially due to overfitting).

- When new input variables (predictors) are added to the model, the R-square is going to increase (because we are adding more variance to the data) independent of an actual performance increase in model. It never decreases when new input variables are added. Therefore, a model with many input variables may seem to have a better performance jsut because it has more input variables. 

## References
- [Evaluation for Regression Models in Machine Learning](https://python.plainenglish.io/evaluation-for-regression-models-in-machine-learning-fb902988c6d6)