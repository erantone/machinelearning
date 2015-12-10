################################################################################################
#     - Naive Bayes implementation based on the book Machine Learning by Tom M. Mitchell
#     - Application to the task of sentinment analysis with text classification using the movie_reviews database from nltk.corpus
#     Copyright (C) 2015  Eric Aislan Antonelo

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################################

from nltk.corpus import movie_reviews
import nltk
import random
import re
import numpy as np

documents = [(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#  0.92 0.88

##  
print 'starting..'
# calcula as palavras mais frequentes
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

def filter_words(words):
    words = filter(lambda x: (len(x) >= 3 and re.search('^\W', x) == None), words)
    return [word.lower() for word in words]

# escolhe como vocabulario as 2000 palavras mais frequentes nos textos
# vocabulary = list(all_words)[:2000]
vocabulary = filter_words(list(all_words)[100:10000])
documents = [(filter_words(doc), t) for (doc, t) in documents ]
train_set, test_set = documents[100:], documents[:100]
train_set = documents  #documents[100:], documents[:100]
targets = ['pos', 'neg']
n_v = len(vocabulary)

##  
print '...'
def train_naive_bayes(train_set):
    P_v = {}
    P_word = {}
    docs = {}
    Text = {}
    for target in targets:
        docs[target] = filter(lambda (example): example[1] == target, train_set)
        P_v[target] = len(docs[target]) / float(len(train_set))
        Text[target] = ' '.join([' '.join(doc) for (doc, t) in docs[target]])
        Text[target] = Text[target].lower()
        n = len(set(Text[target].split() )) # len(docs[target]) #
        P_word[target] = np.zeros(n_v)
        for k in range(n_v):
            if np.mod(k, 1000) == 0:
                print target, k
            wk = vocabulary[k]
            nk = Text[target].count(wk)  # [(wk in set(doc[0])) for doc in docs[target]]   # 
            # nk = len(filter(lambda x: x, nk))
            # print 'nk', nk
            # P_word[target][k] = (nk)/float((n))
            P_word[target][k] = (nk + 1)/float((n + n_v))
    return (P_v, P_word)

print 'Starting training...' 
P_v, P_word = train_naive_bayes(train_set)
print 'Finished training...' 

##  
print 'Starting test...' 
def classify_naive_bayes(doc, P_v, P_word):
    word_positions = []
    for k in range(len(vocabulary)):
        wk = vocabulary[k]
        if wk in set(doc):
            word_positions.append(k)
    likelihood = {}
    print word_positions
    for j in range(len(targets)):
        target = targets[j]
        # likelihood[j] = P_v[target] * reduce(lambda x,y: x*y, [P_word[target][k] for k in word_positions] )
        likelihood[j] = np.log(P_v[target]) + np.sum([np.log(P_word[target][k]) for k in word_positions])
    print likelihood
    jmax = np.argmax(likelihood.values())
    return targets[likelihood.keys()[jmax]]

n = len(test_set)
n_correct = 0
desired = []
output = []
for j in range(n):
    t = classify_naive_bayes(test_set[j][0], P_v, P_word)
    desired.append(test_set[j][1])
    output.append(t)
    if t == test_set[j][1]:
        print t, test_set[j][1]
        n_correct += 1
    else:
        print 'err', t, test_set[j][1]

print n_correct, n_correct/float(n)

## 
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot #, subplot, vlines
plot(P_word['neg'], '.')
plot(P_word['pos'], 'r*')
plt.show()

## 

P_v = {}
P_word = {}
docs = {}
Text = {}
n_v = len(vocabulary)
nks = {'pos': [], 'neg': []}
for target in targets:
    docs[target] = filter(lambda (example): example[1] == target, train_set)
    P_v[target] = len(docs[target]) / float(len(train_set))
    Text[target] = ' '.join([' '.join(doc) for (doc, t) in docs[target]])
    Text[target] = Text[target].lower()
    n = len(set(Text[target].split() ))
    P_word[target] = np.zeros(n_v)
    for k in range(n_v):
        if np.mod(k, 1000) == 0:
            print target, k
        wk = vocabulary[k]
        nk = Text[target].count(wk)
        nks[target].append(nk)
        # P_word[target][k] = (nk)/float((n))
        P_word[target][k] = (nk + 1)/float((n+n_v ))

## 
plot(nks['neg'], '.')
plot(nks['pos'], 'r*')
plt.show()
