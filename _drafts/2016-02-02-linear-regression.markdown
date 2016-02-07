---
layout: post
title:  "Linea"
date: 2016-01-31 23:28:57 -0500
categories: algorithms R 
---


Regression is not just the well known linear regression, but rather, it is about finding a function f(X) for a given X such that f(X) \in \mathbb{R}. In other words we seek to find a function which summarizes or represents X in some way. So, for linear regression f(x) = mx + b and this would model a linear assumption of the data. It could just as well be be a classification problem. Given (X, Y) where Y \in \{-1, 1\}, f(X) could be a majority vote of it’s k nearest neighbors:
eq

where c(x) outputs the class of vector x. We see that this also satisfies our definition of regression. In this post we will focus on linear regression and how to attack the problem by different means.

Here we will use housing data to show different ways to implement linear regression. Below is a scatterplot of the data, which seems to indicate a linear trend.
housing

Python’s scikit-learn provides a easy way to implemt linear regression.
1
2
3
4

from sklearn import linear_model
model = linear_model.LinearRegression()
fit = model.fit(x, y)
np.array([fit.intercept_, fit.coef_]).flatten()

The output for this is
1

array([ 71270.49244873,    134.52528772])

in other words y = 134.52x + 71270.49.

We should be able to confirm these results by calculations on our own. We can write our linear equation in matrix notation, where \beta_0 = 1: Y = X\beta. We need to solve for \beta. Note however that for X to be invertable it needs to be a square matrix.
\begin{array}{rcl} Y &=& X\beta\\ X^{T}Y &=& X^{T}X\beta\\ (X^{T}X)^{-1}X^{T}Y &=& (X^{T}X)^{-1}X^{T}X\beta\\ (X^{T}X)^{-1}X^{T}Y &=& \beta \end{array}
We now have a formula to solve this arithmetically.
Matrix Solution (click to expand)

We see that this is the same solution.

This regression can also be thought of as a learning problem. In other words can we learn \beta through a process? In order to do this we need to define an error function or loss function. A common loss function J(\beta) is the root mean square error
\begin{array}{rcl} J(\beta) &=& \sqrt{\frac{1}{p}\sum\limits_{i=1}^{n}(Y - \hat{Y})^2}\\ &=& \sqrt{\frac{1}{p}\sum\limits_{i=1}^{n}(Y - X_i\hat{\beta})^2} \end{array}

where X \in \mathbb{R}^{n\times p}, X_i is the ith row of X, and \hat{\beta} is an estimate for \beta. We seek to minimize this function. By doing so the estimates \hat{\beta} approach \beta. A popular optimization algorithm is stochastic gradient descent, which updates the estimates for \hat{\beta} with each X_i. The code below will output our learned \beta for a given input X \text{ and } Y.
Stochastic Gradient Descent (click to expand)

One nuance here is that learning rate \alpha in the update formula is a factor along with X, so if X is huge \alpha would need to be minimized so we do not make too large of a descent. In order to avoid this we normalize the data by taking its Z-score (zero mean and unit standard deviation). This will produce a \beta on normalized data. We want to output a \beta that may be used with un-normalized input data. Here let the hat variables be normalized and the non-hat variables be non-normalized. We need solve \hat{X}\hat{\beta}= X\beta for \beta. Typically we would left multiply by X^T and take the inverse to solve for \beta (similar to above), but in our case the first column is all ones the determinate of X^TX=0 which indicates it is singular and non-invertable. Due to the special case of our data we do a little trick by multiplying through by X instead of X^T and still solve for \beta.
\begin{array}{rcl}\hat{X}\hat{\beta} &=& X\beta\\ X\hat{X}\hat{\beta} &=& XX\beta\\ (XX)^{-1}X\hat{X}\hat{\beta} &=& (XX)^{-1}XX\beta\\ (X^2)^{-1}X\hat{X}\hat{\beta} &=& \beta \end{array}

This formula will transform \beta back into its original domain. The output for the learned parameters are very close to what was solve by linear algebra above.
1
2

linear_regression_stoc_grad_desc(x, y)
    (71879.625007811293, 134.17150367460727)

This approach could potentially be useful when, for example, your dataset does not fit in main memory. It also give insight how an optimization problem can be an alternative to an analytic solution.

source code: Linear Example

