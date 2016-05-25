---
layout: post
title:  "Simple Autoencoder"
date: 2016-03-05 23:28:57 -0500
categories: algorithms R 
---

This Theano code below is an implementation of the Autoencoder example in the Tom Mitchell book Machine Learning.
The input for the model is a one-hot encoding of the numbers one through eight. The bottleneck of the model
is of dimension three. The really interesting thing about this is that what the Autoencoder learns is a base-2 
representation of these numbers. The very last part of the code just prints out the learned features in 
base 10. The numbers are not in order, the network did not learn the concept of "order," but there is a 
1-to-1 correspondence between the input and the compressed representation---the learned parameters. 

{% highlight python %}
import numpy as np
import theano
import theano.tensor.nnet as nnet
import theano.tensor as T
{% endhighlight %}

I find it a bit disorienting when neural network libraries define a layer and they ask for the input and
output dimensions. This leads to, for example, a 3-layer network and two "layer" objects (Keras does this).
I think it is more natural to define the connections, the weight matrix between the layers, that is what the
net definition is here. 

{% highlight python %}
def net(w, x):
    b = np.ones((1)).astype(theano.config.floatX)
    x_pad = T.concatenate([b, x])
    return nnet.sigmoid(T.dot(w.T, x_pad))


def update(theta, cost):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))


# +1 for the bias
net_init = lambda in_dim, out_dim: theano.shared(np.random.rand(in_dim+1, out_dim).astype(theano.config.floatX))
{% endhighlight %}

Here we define the encoder and decoder. Input and ouput are of dimension 8 and the bottleneck is dimension 3.
{% highlight python %}
x = T.dvector()
inputs = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                   [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]) 

theta1 = net_init(8, 3)
theta2 = net_init(3, 8)

encoder = net(theta1, x) 
decoder = net(theta2, encoder) 

delta = decoder - x
fc = T.dot(delta, delta)

cost = theano.function(inputs=[x], outputs=fc, 
                       updates=[(theta1, update(theta1, fc)),
                                (theta2, update(theta2, fc))])

get_code = theano.function(inputs=[x], outputs=[encoder])

cur_cost = 0
for i in range(15000):
    for k in range(len(inputs)):
        _cost = cost(inputs[k]) 
    if i % 1000 == 0: 
        print('cost: %s' % (_cost))

print('\n// base 2 to base 10 conversion //')
for i in range(8):
    arr=(get_code(inputs[i])[0]>=.5).astype('int32')
    print(4*arr[0]+2*arr[1]+1*arr[2])
{% endhighlight %}

    cost: 4.4972371593574625
    cost: 0.13448174879634833
    cost: 0.06545246847573791
    cost: 0.0326305660039747
    cost: 0.02147447430168852
    cost: 0.015976155128160183
    cost: 0.012697288874943261
    cost: 0.010519634481520525
    cost: 0.00896935100774384
    cost: 0.007810473057166605
    cost: 0.006912102565645862
    cost: 0.006195762886493091
    cost: 0.005611552097397281
    cost: 0.005126232732349696
    cost: 0.004716819751639709
    
    // base 2 to base 10 conversion //
    1
    0
    6
    5
    3
    7
    4
    2

So we see that the network learned to represent the input in a 3-dimensional space.