---
layout: post
comments: true
title:  "Convolutional Neural Network for Sentence Classification"
date: 2016-08-14 02:28:57 -0500
categories: neural_network
---


###Overview

This is an implementation of the convolutional neural network detailed
in Yoon Kim's [paper][ref] *Convolutional Neural Networks for Sentence
Classification.* The [associated code][chewning_code] can be found on
github. Here we implement binary sentiment classification of the Movie
Review (MR) dataset in [Lasgne][ref]. The network parameters are similar
to what is indicated in the paper, namely
 
* word embeddings with random initialization, non-static,
* embedding size 128,
* word windows of 3, 4, and 5,
* single channel,
* 128 convolutional filters,
* batch size 50, and
* dropout 50%.

There are a few other known implementations:

* The author's [implementation][ref], written in Theano. Reported accuracy is 76.1%.
* Denny Britz' [implementation][ref], written in TensorFlow. Reported accuracy is 76%.
* HarvardNLP [implementation][ref], written in Torch. Reported accuracy is 75.9%.
* Alexander Rakhlin's [implementation][ref], written in Keras. Reported accuracy is unknown.

After 25 epochs, the accuracy of this implementation is 76.8%, which is slightly better than
what is reported in the original paper.

While implementing this network I was able to achieve an accuracy of 76% but there were bugs
in the implementation. Testing the network help greatly. Details on testing can be found
[here][cnn_nlp]. So, even a network with implementation errors performed well. Fixing these
bugs help increase the accuracy another 0.7%. 

###References {% include anchor.html name="references" %} 
* [[1] Convolutional Neural Networks for Sentence Classification, (Kim)][kim]
* [[2] Yoon Kim's Theano implementation][kim_code]
* [[3] Denny Britz' TensorFlow implementation][britz_code]
* [[4] HarvardNLP Torch implementation][harvard_code]
* [[5] Alexander Rakhlin's Keras implementation][rakhlin_code]
* [[6] Lasagne implementation][chewning_code]
* [[7] Lasagne][lasagne]

{% include comments.md page-identifier="cnn_sentence_classification" %} 

[kim]: http://arxiv.org/abs/1408.5882
[kim_code]: https://github.com/yoonkim/CNN_sentence
[britz_code]: https://github.com/dennybritz/cnn-text-classification-tf
[harvard_code]: https://github.com/harvardnlp/sent-conv-torch
[rakhlin_code]: https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
[chewning_code]: https://github.com/init-random/neural_nlp/tree/master/sentence_classification
[lasagne]: http://lasagne.readthedocs.io/ 
[cnn_nlp]: /neural_network/2016/07/25/CNN_NLP.html
[ref]: #references

