---
layout: post
title:  "Linear Regression"
date: 2016-02-07 23:28:57 -0500
categories: regression
---

This is a scatterplot of the Oregon housing [data][housing_data] provided in the Andrew Ng OpenClassroom Machine Learning
course, which shows housing price as a function of living square feet. 

{% include img.html img-src="housing_scatter.png" %} 

Visually, it appears that there is a linear relationship in this data. It would be logical to
fit a linear regression model to this data to learn the relationship between price and the number
of square feet of living area. This can be solved analytically (see Hastie, et al. 2009). In matrix notation this would be

$$\hat{Y} = X \hat{\beta} = X(X^T X)^{-1} X^T Y$$

where $$\hat{Y}$$ is the regression value at $$x (\in X)$$ and $$\hat{\beta}$$ are the learned parameters;
to simplify the notation, we let $$\beta_0$$ be the intercept and $$x_0 = 1$$.
While this will provide an exact solution it is necessary to calculate a matrix inverse, which
may be expensive ($$O(n^3)$$ with Gaussian Elimination) if the number of parameters is large. Let's
look into possible alternatives for learning $$\hat{\beta}.$$ For reference, we will note that
the intercept ($$\beta_0$$) is 71270.5 and slope 134.5.

### Gradient Descent

Let us define an error function or loss function. A common loss function
$$J(\hat{\beta})$$ is the mean square error 

$$
  \begin{array}{rcl} J(\hat{\beta}) &=& \frac{1}{p}\sum\limits_{i=1}^{n}(Y - \hat{Y})^2\\ 
                                    &=& \frac{1}{p}\sum\limits_{i=1}^{n}(Y - X_i\hat{\beta})^2 \\
                                    &=& \frac{1}{p} (Y - X\hat{\beta})^T (Y - X\hat{\beta}) \\ 
                                    &\propto& (Y - X\hat{\beta})^T (Y - X\hat{\beta}).  
  \end{array}
$$

The penultimate step is just in matrix notation and the last step is justified in that if we minimize
a function, then we can also minimize a function proportional to it without loss of generality.



A common first step in trying to understand the relationship between these variables
would be to fit a regression model, which would provide some insight into how house
price is related to square foot of living space. In particular, this data appears to
have linear relationship. Linear regression models have an analytic solution, which
for completeness we show here.

$$
  \begin{eqnarray}
                      Y &=& X \beta \\
                  X^T Y &=& X^T X \beta  \\
     (X^T X)^{-1} X^T Y &=& (X^T X)^{-1} (X^T X) \beta \\
     (X^T X)^{-1} X^T Y &=& \beta 
  \end{eqnarray}
$$

This implies that $$Y = (X^T X)^{-1} X^T \beta$$

Regression is not just the well known linear regression, but rather,
it is about finding a function f(X) for a given X such that f(X) \in
\mathbb{R}. In other words we seek to find a function which summarizes
or represents X in some way. So, for linear regression f(x) = mx +
b and this would model a linear assumption of the data. It could just
as well be be a classification problem. Given (X, Y) where 
$$Y \in \{-1, 1\}, f(X)$$ could be a majority vote of it’s k nearest neighbors: eq

where c(x) outputs the class of vector x. We see that this also satisfies
our definition of regression. In this post we will focus on linear
regression and how to attack the problem by different means.

Here we will use housing data to show different ways to implement linear
regression. Below is a scatterplot of the data, which seems to indicate
a linear trend.
housing

Python’s scikit-learn provides a easy way to implemt linear regression.

from sklearn import linear_model model = linear_model.LinearRegression()
fit = model.fit(x, y) np.array([fit.intercept_, fit.coef_]).flatten()

The output for this is array([ 71270.49244873,    134.52528772])

in other words y = 134.52x + 71270.49.















We should be able to confirm these results by calculations on our own. We
can write our linear equation in matrix notation, where \beta_0 = 1:
Y = X\beta. We need to solve for \beta. Note however that for X to
be invertable it needs to be a square matrix.  \begin{array}{rcl}
Y &=& X\beta\\ X^{T}Y &=& X^{T}X\beta\\ (X^{T}X)^{-1}X^{T}Y &=&
(X^{T}X)^{-1}X^{T}X\beta\\ (X^{T}X)^{-1}X^{T}Y &=& \beta \end{array}
We now have a formula to solve this arithmetically.  Matrix Solution
(click to expand)

We see that this is the same solution.

where X \in \mathbb{R}^{n\times p}, X_i is the ith row of X,
and \hat{\beta} is an estimate for \beta. We seek to minimize this
function. By doing so the estimates \hat{\beta} approach \beta. A popular
optimization algorithm is stochastic gradient descent, which updates the
estimates for \hat{\beta} with each X_i. The code below will output our
learned \beta for a given input X \text{ and } Y.  Stochastic Gradient
Descent (click to expand)

One nuance here is that learning rate \alpha in the update formula
is a factor along with X, so if X is huge \alpha would need to be
minimized so we do not make too large of a descent. In order to
avoid this we normalize the data by taking its Z-score (zero mean
and unit standard deviation). This will produce a \beta on normalized
data. We want to output a \beta that may be used with un-normalized
input data. Here let the hat variables be normalized and the non-hat
variables be non-normalized. We need solve \hat{X}\hat{\beta}= X\beta
for \beta. Typically we would left multiply by X^T and take the inverse
to solve for \beta (similar to above), but in our case the first column
is all ones the determinate of X^TX=0 which indicates it is singular and
non-invertable. Due to the special case of our data we do a little trick
by multiplying through by X instead of X^T and still solve for \beta.
\begin{array}{rcl}\hat{X}\hat{\beta} &=& X\beta\\ X\hat{X}\hat{\beta}
&=& XX\beta\\ (XX)^{-1}X\hat{X}\hat{\beta} &=& (XX)^{-1}XX\beta\\
(X^2)^{-1}X\hat{X}\hat{\beta} &=& \beta \end{array}

This formula will transform \beta back into its original domain. The
output for the learned parameters are very close to what was solve by
linear algebra above.

linear_regression_stoc_grad_desc(x, y)
    (71879.625007811293, 134.17150367460727)

This approach could potentially be useful when, for example, your dataset
does not fit in main memory. It also give insight how an optimization
problem can be an alternative to an analytic solution.

source code: Linear Example


[housing_data]: http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html

