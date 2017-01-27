---
layout: post
comments: true
title:  "Language Models"
date: 2017-01-27 23:28:57 -0500
categories: language models probability
---

The full code from this article can be found [here [4]][ref] 

In this article we will cover what language models (LM) are and how LMs can provide a probabilistic framework as a basis for 
many natural language workflows. This LM will not be state of the art, rather it is meant to provide an introduction
to the topic and to show some common features associated with LMs---smoothing and evaluation. The smoothing
we will show here is Jelinek and Mercer trigram smoothing, which uses the [EM algorithm [1]][ref]. The EM algorithm is
an underutilized method in machine learning and this will provide an opportunity to show one of its many uses. 

In its most basic form, an LM asks either

$$p(w_1, \ldots, w_n)$$

or 

$$p(w_n | w_1, \ldots, w_{n-1}).$$

In other words, we would like to know the probability of a sequence of words or the probability of a word given the words that precede it---its history or context. Upon first inspection it does not appear that we have gained much by stating these language questions in a probabilistic manner. There are, however, a number of applications that are facilitated by being able to answer these questions. 

 * Spelling correction: $$p(the\, store) > p(teh\, store)$$
 * Speech recognition: $$p(i\, love\, espresso) > p(i\, of\, express\, so)$$
 * Automated translation: $$p(strong\, winds) > p(power\, winds)$$ given $$starke\, winde$$
 * Text completion: suggesting the next word, e.g. instant messaging
 * Generative models: create text documents
 * Word embeddings: one "side effect" of LMs in neural networks 
 
As you can see, being able to answer these questions well would greatly help in practical situations. As stated, LMs allow us 
to assign probabilities to words. This may seem rather unintuitive; for example, what the probability of the sequence 
$$firefox\, browser$$? We may not readily know the answer to this question, but it would make sense that however we assign 
a probability it should be greater than $$p(carpet\, chipmunk)$$ which would be a rare phrase.  

A first step in developing a probabilistic model would be to do the following.

$$p(meeting | what\, time\, is\, the) = \frac{\#(what\, time\, is\, the\, meeting)}{\#(what\, time\, is\, the)}$$

where $$\#(\cdot)$$ is the number of times we saw the phrase in a document. This equality is called the maximum likelihood estimate (MLE). 
The MLE is calculated by determining the count of the phrase (numerator) from our training data and normalizing this value into a 
probability (the denominator). You should convince yourself by working out a simple example that this is what this equality 
indicates. Now, "what time is the meeting" might seem like a common enough phrase, but we could just as well have 
seen "what time is the exam" which might be less common. It is unlikely that we would be able to account for every 
possible sentence variation in our training data. Because of this, we cannot assign probabilities to all possible phrases. 
So, this solution will not scale.

It could be the case that we do not need the full context of the sentence preceding the word in order to make a valid 
prediction. Maybe using just a few preceding words could be sufficient? This is called the Markov Assumption and if we 
accept this simplification it will allow us to write probabilities of the form

$$p(meeting | is\, the)$$

where we only consider a short window as the preceding context, two words in this case. A context of two words is 
called a _trigram model_ as there are three words used in total. It is now much more likely 
that we may see these shorter phrases in the training data. Before we look at how the Markov Assumption is utilized, let's 
take a look at these probabilities in terms of the Chain Rule. From Bayes rule, for any two events $$A$$ and $$B$$, we have the equality

$$p(AB) = P(B|A)p(A)$$

where $$p(AB)$$ is the joint probability of both $$A$$ and $$B$$ occurring. To utilize the chain rule, if we had the joint 
probability $$p(ABC)$$ then

$$p(ABC) = p(C|AB)p(AB) = p(C|AB)p(B|A)p(A)$$

which is just an extension of the first equality. Now if we use a bigram Markov Assumption then we see that

$$
\begin{align*}
   p(what\, time\, is\, the\, meeting) &= p(meeting|what\, time\, is\, the)p(the|what\, time\, is)p(is|what\, time)p(time|what)p(what)\\
     &\approx p(meeting|the)p(the|is)p(is|time)p(time|what)p(what) 
\end{align*}
$$

which provides an approximation to the true value, but what we loose in accuracy we gain in generality. In this manner
we are better able to avert the issues of scale. Given the Markov Assumption we may not have seen "what time is the exam"
in out training data, but it may be more likely, given less context, that may observe "is the exam."
This is the basis 
for many language models. Now let's take a look at an initial application of this simple model. We will train the model on 
Franz Kafka's [The Trial [2]][ref] and use a generative model to create new text that is similar in style to this novel.

First we import some needed classes and define some utility methods to generate training and test sentences. Sentences are 
padded with _xsb_ and _xse_ which are sentinel values to demarcate beginning and end of sentences. 


{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer
from spacy.en import English
import numpy as np
import functools


def batch_until(lines, condition=''):
    # batch paragraphs
    batch = ''
    for l in lines:
        _l = l.strip()
        if _l == condition and batch != '':
            yield batch
            batch = ''
        if _l == condition:
            continue
        batch += ' ' + _l


def sentence_gen(lines):
    # yield sentences
    for doc in parser.pipe(batch_until(lines), batch_size=10, n_threads=2):
        for s in doc.sents:
            # sentinel values for begin/end sentence
            yield '%s %s %s' % ('xsb xsb', s.text.strip(), 'xse')
            
            
def tst_idx(tst_share=.05):
    # indexes for test data
    return np.random.choice(100, int(100*tst_share), replace=False)
    
    
def trn_tst_split_sentence_gen(filename, tst_idx=None):
    # yeild train/test data
    with open(filename, 'r') as lines:
        for idx, s in enumerate(sentence_gen(lines)):
            if tst_idx is not None and (idx%100) in tst_idx:
                yield s
            elif tst_idx is not None:
                continue
            else:
                yield s


the_trial = './data/the_trial.txt'

# sentence recognition can be non-trivial, so use spaCy
parser = English()

trn_sentence_gen = trn_tst_split_sentence_gen(the_trial)

tstidx = tst_idx()
{% endhighlight %}

Next we create a word count matrix of the training data. 

{% highlight python %}
window_size = 3
ngram = CountVectorizer(strip_accents='ascii', ngram_range=(1, window_size), 
                        lowercase=True, token_pattern='(?u)\\b\\w+\\b')

ngram_dtm = ngram.fit_transform(trn_sentence_gen)

ngram_vocab = np.array(ngram.get_feature_names())
ngram_word_count = np.array(ngram_dtm.sum(axis=0))[0]

words = [w for w in ngram.vocabulary_.keys() if ' ' not in w]

vocab_size = len(words)

_bp = ngram.build_preprocessor()
_bt = ngram.build_tokenizer()
# data transformer for test data
sentence_transformer = lambda s: _bt(_bp(ngram.decode(s)))
{% endhighlight %}

This method will provide the probability of a phrase relative to the training data. 

{% highlight python %}
ngram_count = lambda w: vocab_size if w=='' else 0.0 if ngram.vocabulary_.get(w) is None else ngram_word_count[ngram.vocabulary_.get(w)] 

def ngram_prob(ngram_str, n_words_doc):
    if ngram_str == '':
        # 0-gram
        return 1/n_words_doc
    conditioning = ngram_str.split()[:-1]
    numer = ngram_count(ngram_str)
    if len(conditioning) == 0:
        # uni-gram
        return numer/n_words_doc
    conditioning = ngram_count(' '.join(conditioning))
    if conditioning == 0.0:
        # out of vocabulary
        return conditioning
    # n-gram
    return ngram_count(ngram_str)/conditioning
{% endhighlight %}



The iteration below is the EM algorithm, so let's try and understand what is going on here. First, we create 
a trigram model of the following form:

$$\tilde{p}(C|AB) = \lambda_0 p(C|AB) + \lambda_1 p(C|A) + \lambda_2 p(C) + \lambda_3 p(V)$$

which is subject to

$$\Sigma_j  \lambda_j = 1.$$

So, this is an interpolated estimate of the maximum likelihood trigram model that we saw above. Here 

$$p(V) = 1/\#\{words\, in\, document\}$$

so we can provide a probability for out of vocabulary words. With this interpolated model even it we did not
see "is the exam" in our training data we will still be able to provide an estimate if we only observed, e.g. "exam."
This interpolated model is one example of language _smoothing_, which allows for a more robust representation 
of our data. Now the question remains, how do we estimate the lambdas? We will do this with the EM algorithm.

EM may be used to find implicit values in data that were not overtly observed. For example, EM could be used to
find the means of a multi-modal distribution without knowing these values _a priori_. EM stands for _Expectation Maximization_
and it uses a two-step process. In our use case the expectation will be provide values for the n-grams using the maximum likelihood
estimate. For example, if the trigram was not seen the its estimate would be zero, similarly we would estimates for the other
 n-grams. Given the expectation we maximize the lambdas, which in this case is to just
 normalize the lambdas so they sum to one. This two-step process is iterated over until the stopping condition, 
 which here we hardcode to five iterations.
 
Notice below that the lambdas below are tuned on the test data. This is important to use a separate set of data
because of used the original data the expectation of the trigram would be one and the other values would be zero.
We need the test data to generalize to unforeseen phrases and words. 


{% highlight python %}
n_words_doc = functools.reduce(lambda a, b: a+b, [ngram_word_count[ngram.vocabulary_.get(w)] for w in words])

# [('xsb xsb now', 'xsb now', 'now', ''),
#  ('xsb now they', 'now they', 'they', ''), ...]
to_ngrams = lambda s: [tuple(' '.join(s[(idx-3+_idx):idx]) for _idx in range(4)) for idx in range(3, len(s)+1)]

# [[0.010, 0.005, 0.065, 0.000], ...]
to_ngram_vals = lambda ngrams: [[ngram_prob(ng, n_words_doc) for ng in ngs] for ngs in ngrams]

# ls: initial lambda values
ls = np.array([.25,.25,.25,.25])

# 5 iterations over EM
for _ in range(5):
    # ngram values
    ng_arr = np.zeros(4)
    for str_hld in trn_tst_split_sentence_gen(the_trial, tstidx):
        transformed = sentence_transformer(str_hld)
        ngrams = to_ngrams(transformed)        
        ngvals = to_ngram_vals(ngrams)
        # Expectation
        expctations = [ls*ngval for ngval in ngvals]
        for e in expctations:
            # z normalizing constant
            z = e.sum()
            ng_arr += e/z

    # Maximization        
    ls = ng_arr/ng_arr.sum()
    print(ls)

# lambda values
#    [  7.72111746e-01   1.89254926e-01   3.84487082e-02   1.84619248e-04]
#    [  9.27978030e-01   6.83397556e-02   3.68208718e-03   1.26911721e-07]
#    [  9.76312353e-01   2.31480939e-02   5.39552881e-04   4.86541737e-10]
#    [  9.91856508e-01   7.87760196e-03   2.65890093e-04   1.18089926e-11]
#    [  9.96941497e-01   2.81612745e-03   2.42375676e-04   5.77755587e-13]

{% endhighlight %}

Now that we have our lambdas and interpolated trigram model let's put it to use by generating some text. We trained
on the The Trial by Franz Kafka. The lead character in this novel is identified by "K." So, "k" in the generated 
sentences below are not a mistake, but refers to the character. No punctuation is inserted here, so for instance we see
"that s" which was treated as two separate words but the model determined that they should be adjacent to each
other to indicated the single word "that's." The model is not perfect it can certainly create run-on sentences as
seen in the fourth sentence below, but in general the text reads in a similar manner to the novel.

{% highlight python %}
def gen_word(context):
    r = np.random.random()
    acc_sum = 0.0
    for w in words:
        snt = '%s %s' % (context, w)
        transformed = sentence_transformer(snt)
        ngrams = to_ngrams(transformed)        
        ngvals = to_ngram_vals(ngrams)[0]
        acc_sum += (ls*ngvals).sum()
        if acc_sum>=r:
            if w == 'xsb':
                continue
            if w == 'xse':
                return '.'
            return w

def gen_sentence():
    context = ['xsb', 'xsb']
    sent = []
    while True:
        w = gen_word(' '.join(context))
        if w == context[1]:
            continue
        sent.append(w)
        context[0] = context[1]
        context[1] = w
        if w == '.':
            return ' '.join(sent)


for _ in range(5):
    print(gen_sentence() + '\n')

# SAMPLE GENERATED SENTENCES
#    
#     that s going to happen again she agreed and smiled at k .
#     
#     i ve only been talking about the entrance .
#     
#     now then josef he then called across to it he said that as far as could be .
#     
#     it was also a man could fall through but it was a very important once you ve worked hard to remember every tiny 
#     action and event from the shock of being admitted in the process is that of course need to consider that the 
#     lawyer he had acted with no result .
#    
#     he might do anything about the trial too .
{% endhighlight %}


Without going into too much detail here, it is common that we would want to evaluate our model in some way. A common
method is to use the measure of [Perplexity [3]][ref] for the model. The perplexity of our model is about 4.70, which 
can be interpreted to mean that on average the model is confused on selection of a word by about five words. In other
words, the model might be prone to mis-select the perfect next word by randomly choosing between five words. This
perplexity value is quite low, but this is due to the narrow distribution of the training, i.e. it is not random
data from disparate sources like the web.

{% highlight python %}
tst_n = 0 # length normalization
accum_prob = 0
for str_hld in trn_tst_split_sentence_gen(the_trial, tstidx):
    tstrsplt=sentence_transformer(str_hld)
    ngrams=[tuple(' '.join(tstrsplt[(idx-3+_idx):idx]) for _idx in range(4)) for idx in range(3, len(tstrsplt)+1)]
    tst_n += len(ngrams)
    ngvals=[[ngram_prob(ng, n_words_doc) for ng in ngs] for ngs in ngrams]
    x = [ls*ngval for ngval in ngvals]
    for _x in x:
        accum_prob -= np.log(_x.sum())
       
np.exp(accum_prob/tst_n)

# PERPLEXITY
#    4.7024638719690595
{% endhighlight %}


### Conclusion

Here we looked at Language Models and how they can be used to assign probabilities to words and phrases. We
used an interpolated trigram model whose lambda values were learned by the EM algorithm. We then used the model 
to train some data and generate text similar to the training data. Finally, we briefly looked at how you
can evaluate and compare models based on their perplexity value.


###References {% include anchor.html name="references" %} 
* [[1] The Trial by Franz Kafka, Project Gutenberg][kafka]
* [[2] EM Algorithm Introduction][em]
* [[3] Perlexity][pp]
* [[4] Source Code][code]

{% include comments.md page-identifier="language_models" %} 

[kafka]: http://www.gutenberg.org/ebooks/7849
[em]: https://github.com/init-random/analytics-notebooks/blob/master/src/em_algorithm.ipynb 
[pp]: https://web.stanford.edu/~jurafsky/slp3/4.pdf
[code]: https://github.com/init-random/analytics-notebooks/blob/master/src/language_model.ipynb
[ref]: #references
