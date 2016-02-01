---
layout: post
title:  "Distribution Sampling"
date: 2016-01-31 23:28:57 -0500
categories: algorithms R 
---

Algorithms such as [AdaBoost][adaboost] impose a distribution over the dataset, i.e. there is a weight assigned to each record in the training data. Descriptions of these algorithms typically do no indicate how this weighted selection is actually done.

To generalize this problem, the question we are asking is, given a arbitrary distribution of items $$i$$, where $$i$$ has weight $$w_i$$, how can we sample from this distribution subject to these weights? If $$j$$ has weight $$w_j=1$$ and $$k$$ weight $$w_k=2$$, then we would expect $$k$$ to be independent randomly selected about twice as often as $$j$$. More concretely, a die is uniformly distributed. If we sampled from this distribution 60 times, we would expect to see each side of the die about 10 times. If however, the die was biased, how can we sample from this distribution, i.e. how can we *simulate* rolling the die?

{ % include image.html url="/images/my-cat.jpg" description="My cat, Robert Downey Jr." %}

From here on we assume that the the weights are normalized, that is

$$\hat{w_i} = \frac{w_i}{\sum_{j=1}^{n}w_j}.$$

We cannot directly select an item subject to the distribution. We can however do so indirectly. Random number generators can easily provide uniform distribution random numbers. So, this is what we will use. The following formula indicates how this may be leveraged.

$$\arg\min_{k} \sum_{i=1}^{k}w_i \ge r$$

Here, r is a uniform random number and the summation is the cumulative sum of the weights. The argmin indicates that we seek the first record whose cumulative weighted sum is greater than or equal to r. In this way we can leverage a uniform random number to select an item from an arbitrary distribution.

As an implementation detail, it would be advantageous to memoize the cumulative sum for each record so this would not need to be calculated each time. Also note that the cumulative sum is ordered, so we could use a binary search to find the argmin in question. Here is some source code showing this technique.

 
{% highlight r %}
library(gtools)

id <- 1:100
# pick some weights for an arbitrary distribution
dist <- round(rnorm(100, 100, 27))
# normalize
dist <- dist/sum(dist)
cum.dist <- cumsum(dist)
  
d <- data.frame(list(id=id, dist=dist, cum.dist=cum.dist))
# view distribution
hist(d$dist, breaks=10)
  
# rand select func
dist.select <- function(cum.dist, target) {
  val <- binsearch(function(i) { cum.dist[i] },
    1:length(cum.dist),
    target=target)$where[2]
  if(is.na(val)) val <- 1
  val
}
  
# new selection subject to distribution
new.dist <- sapply(1:100, FUN=function(idx) {
  d$dist[dist.select(d$cum.dist, runif(1))]
})
 
# distribution should be similar to the hist above
hist(new.dist, breaks=10) 
{% endhighlight %}


[adaboost]: https://en.wikipedia.org/wiki/AdaBoost 


