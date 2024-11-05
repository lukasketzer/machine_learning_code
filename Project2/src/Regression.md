
# a) Regression

## 1.
We want to predict a persons weight based on all of the other features.
To determine the performance (?) of our dataset we decided on using 2-cross-validation.

### Layer 1
On the first layer we decided to use K-fold cross-validation with K = 10 (may still change). We chose said methode since it is a very good compromise.
We are able to get performance test on all of our data and we don't need to use as much computational power. Computational power is especialy important since we have realtively big dataset with around 2100 observations. 

### Layer 2
On the second layer we opted to use feature selection to improve the performance of our dataset. 


## 2. Regularization
Bias variabl $\lambda$ 

> for a small $\lambda$ the model will overfit the data
