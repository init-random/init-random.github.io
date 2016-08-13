---
layout: post
comments: true
title:  "Python Pipelines for Sentiment Analysis"
date: 2016-07-18 23:28:57 -0500
categories: sklearn 
---


###Originally posted on <a href="https://blog.answr.com/2016/07/18/python-pipelines-for-sentiment-analysis/"><img src="/images/answr.png" /></a>
{: style="text-align: center;"}

------------------------------------------------
------------------------------------------------

<br />
How many times have you written boilerplate code that transforms your
data for input into an algorithm? Or maybe you are doing preliminary
testing on multiple types of models to test their performance. Python’s
scikit-learn offers an easy way to set up work-flows through their
Pipeline interface, which can greatly simplify data transformation and
model set up. Let’s take a look at some data and see how this can be
implemented in practice.

###Sentiment Data
{% include img.html img-src="smiley.jpg" %} 

In there era of social media and brand reputation management, knowing
the sentiment of your user base relative to your product is vitally
important. Do you have insight into how much people approve of your
product? Kaggle hosts data science competitions and is a great place to
pick up new data for all sorts of problem domains and today we will take
at the Rotten Tomatoes dataset which we will use to create some models
to predict user sentiment. This data is comprised of phrases from movie
reviews that are labeled on a scale ranging from zero to four where zero
indicates a negative review and four indicates a positive review. For
your own projects you can either get publicly available data like this
to train on or you can use manually labeled data, like Tweets, specific
to your particular product. The benefit of using your own data is that
the vocabulary will be more specific to your problem domain. On the
other hand, you will need to invest time manually labeling the sentiment
your data.

We first need to load our data, so here is a helper function so that we
can start training.


{% highlight python %}
import numpy as np

from sklearn import cross_validation
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def load_data(coarse=True, shuffle=True):
    x = []
    y = []
    for record in open('./data/train.tsv'):
        atoms = record.split('\t')
        sentiment = atoms[3][:-1]
        # skip header
        if sentiment == 'Sentiment': continue
        sentiment = int(sentiment)
        pos = sentiment
        if coarse:
            if sentiment == 2: continue
            pos = int(sentiment > 2)
        x.append(atoms[2].lower())
        y.append(pos)
        if shuffle:
            perm = np.random.permutation(range(len(y)))
            x, y = ((np.array(x)[perm]).tolist(), (np.array(y)[perm]).tolist())
    return x, y
{% endhighlight %}

This simply returns a tuple of training data and its associated class
label, i.e. it’s sentiment. The coarse parameter will be explained
later.  

### Pipelines

The input into our model is raw text. We will be using a logistic
regression to classify each phrase, but logistic regression requires
that inputs be numeric and not text. We run the data through two
transformations, CountVectorizer and TfidfTransformer in order to
accommodate for this. The former provides word counts for each phrase
and the latter is a transformation of the word counts to penalize very
common words, giving more priority to “content” words.

{% highlight python %}
x, y = load_data(coarse=False)

count_vectorizer = CountVectorizer()
tfidf = TfidfTransformer()
logistic_regression = LogisticRegression()

model = Pipeline([('counts', count_vectorizer),
                  ('tfidf', tfidf),
                  ('regression', logistic_regression), ])

scores = cross_validation.cross_val_score(model, x, y, cv=5, scoring='accuracy')
print('Accuracy: %0.2f' % (scores.mean()))

# Accuracy: 0.63
{% endhighlight %}

Accuracy is 63%. On the surface of it, that does not sound that
good. Looking into the Kaggle forums for this competition it looks
like a reasonable baseline is around 61% and many people initially
get around 56%. Given that we used all of the default settings for
the models and the only preprocessing was to lowercase the data, 63%
is not that bad. Random guessing would give an accuracy of 20%. This
is granular scale, however. We may have classified the review as a
4 instead of a 3. Both are on the positive side of the scale so the
accuracy may look a little worse than it is. A common thing to do is
to remove the neutral reviews and categorize the remaining either as
positive or negative sentiment. This is what the coarse parameter does
on the data load. Let’s see what that looks like.


{% highlight python %}
x, y = load_data(coarse=True)

count_vectorizer = CountVectorizer(lowercase=True)
rp = random_projection.SparseRandomProjection(random_state=11)
svd = TruncatedSVD(n_components=2500)
tfidf = TfidfTransformer()
logistic_regression = LogisticRegression()

model1 = Pipeline([('counts', count_vectorizer),
                   ('rand_proj', rp),
                   ('logistic', logistic_regression), ])
model2 = Pipeline([('counts', count_vectorizer),
                   ('svd', svd),
                   ('regression', logistic_regression), ])
model3 = Pipeline([('counts', count_vectorizer),
                   ('tfidf', tfidf),
                   ('regression', logistic_regression), ])

vc = VotingClassifier(estimators=[('model1', model1), ('model2', model2), ('model3', model3)], 
                      voting='hard')

for clf, label in zip([model1, model2, model3, vc], 
                      ['model1_random_projections', 'model2_svd', 'model3_tfidf', 'ensemble']):
    scores = cross_validation.cross_val_score(clf, x, y, cv=2, scoring='accuracy')
    print("Accuracy %s: %0.2f" % (label, scores.mean()))

# Accuracy model1_random_projections: 0.85
# Accuracy model2_svd: 0.81
# Accuracy model3_tfidf: 0.86
# Accuracy ensemble: 0.85
{% endhighlight %}

What is going on here? We loaded the data and trained on binary
positive/negative output classes. We then set up a series of models
utilizing different data transformations: random projections, singular
value decomposition (SVD), and term frequency-inverse document frequency
(TF-IDF). The first two models utilize dimensionality reduction
techniques. The third uses TF-IDF, which was used in first, granular
model. The last output here is an ensemble (mixture) of all three models
where we use a max vote for the classification. Ensembles are typically
used for averaging different types of models. For this ensemble we used
the same model but leveraged different data transformations. It is more
common that ensembles used a mixture of different types of models (see
possible alternatives in the next section), but both paths are worth
exploring. It is interesting to note that the model with the highest
accuracy is the TF-IDF model. Many times it is well worth doing simple
things first and than trying more complex transformations. The accuracy of
the model is around 85%, given further tweaking of the model parameters
and data preprocessing we could probably get another 5% increase.

### Enhancements

As stated, we used many of the default parameters of the models. Here are
a few things you could try on your own to further increase the accuracy.


* document preprocessing
    * bi-grams
    * tokenizing
    * stop word removal
    * stemming
*  try different dimensionality reductions for SVD and random projections, i.e. reduce to a k-dimensional dataset
*  optimizing the parameters of the logistic regression, e.g. regularization and solvers
*  try different ensemble methods provided by scikit-learn
*  try other models other than logistic regression, e.g. Naive Bayes or Support Vector Machines

### Conclusion

We took a look at Pipelines in scikit-learn and how these can be used
to assemble models. We took two views in the data classification,
granular and coarse, and fit a few different models. The motivation was
to simplify boilerplate code and to afford the opportunity to easily
swap out different models. We also took a look at simple ensembles and
how these could be used in your work.

{% include comments.md page-identifier="pipelines_for_sentiment" %} 
