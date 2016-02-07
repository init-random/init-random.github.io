---
layout: post
title:  "Distribution Sampling"
date: 2016-01-31 23:28:57 -0500
categories: algorithms R 
---

Algorithms such as [AdaBoost][adaboost] impose a distribution over the
dataset, i.e. there is a weight assigned to each record in the training
data. During the training of this algorithm, these weights are either used
directly in the weak learners or they may be used to take bootstrap samples.
In the latter case we need a way to sample with replacement from our dataset of size $$N$$
subject to the distribution imposed by the weighting. 
Descriptions of these algorithms typically gloss over how this
weighted selection is actually done in practice.

To generalize this problem, the question we are asking is, given a
set of items $$i$$ in dataset $$D$$ of size $$N$$, where $$i$$ is assigned weight $$w_i$$,
how can we sample from this distribution subject to these weights? If
$$j$$ has weight $$w_j=1$$ and $$k$$ weight $$w_k=2$$, then we would
expect $$k$$ to be independent randomly selected about twice as often
as $$j$$. More concretely, a die is uniformly distributed. If we sampled
from this distribution 60 times, we would expect to see each side of the
die about 10 times. If however, the die was biased, how can we sample
from this distribution, i.e. how can we *simulate* rolling the die?

From here on we assume that the the weights are normalized, that is

$$w_i  \leftarrow \frac{w_i}{\sum_{j=1}^{N}w_j}.$$

We cannot directly select an item subject to the distribution. We can
however do so indirectly. Random number generators can easily provide
uniform distribution random numbers. We will leverage this to sample from an arbitrary distribution. 
Let $$f(w)$$ indicate the weight distribution and its cumulative distribution
by $$F(w)$$. We uniformly generate a random number $$r \in [0, 1]$$. Now
we need find $$w$$ such that $$F(w)=r$$, i.e. $$w=F^{-1}(r)$$, which is the inverse
function of $$F$$ and not analytically solvable. For a discrete distribution this may be restated as

$$\arg\max_{k} \sum_{i=1}^{k}w_i \le r.$$

For the implementation, it would be advantageous to memoize the
cumulative sum so this would not need to be calculated
for each sample. Also note that the cumulative sum is monotonic, so we could use
a binary search to find the argmax in question. Below is a generated
multi-modal distribution which we wish to sample.

 
{% highlight r %}
library(gtools)
library(plyr)

# create a multi-modal distribution
d1 <- round(rnorm(100, 50, 25))
d2 <- round(rnorm(100, 100, 15))

dist <- join(count(d1), count(d2), by='x', type='full')

plot(density(rep(dist$x, dist$freq)), col='lightblue', lwd = 3)
{% endhighlight %}

{% include img.html img-src="dist-orig.png" %} 

Note that the modes of this distribution are around 50 and 100. So, if
we sampled a few numbers we would expect them to center around those values.
Below is the sampling code.

{% highlight r %}
# normalize
dist$freq <- dist$freq/sum(dist$freq)
dist <- dist[order(d$x), ]
dist$dist <- dist$x
# cumulative distribution: F(w)
dist$cum.dist <- cumsum(dist$freq)

# rand select func
dist.select <- function(cum.dist, target) {
  val <- binsearch(function(i) { cum.dist[i] },
    1:length(cum.dist),
    target=target)$where[2]
  if(is.na(val)) val <- 1
  val
}
  
# new selection subject to distribution; 100 samples
new.dist <- sapply(1:100, FUN=function(idx) {
  d$dist[dist.select(d$cum.dist, runif(1))]
})
 
# distribution should be similar to the hist above
plot(density(new.dist), col='orange', lwd = 3)
{% endhighlight %}

Below is the distribution obtained from the sampling. You can see
that even with only 100 samples, the original and sampled distributions
are quite similar. 

{% include img.html img-src="dist-sample.png" %} 


[adaboost]: https://en.wikipedia.org/wiki/AdaBoost 


