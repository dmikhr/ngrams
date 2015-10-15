# -*- coding: utf-8 -*-

from sys import argv
import operator

from sklearn.feature_extraction.text import CountVectorizer

HELP_MSG = \
"""
USAGE:

$ python ngram.py file_with_text number_of_words

"""

fname = argv[1]
N = int(argv[2])

text = open('data/%s.txt' % fname, 'r').read()

v = CountVectorizer(ngram_range=(N,N))

X = v.fit_transform([text])

results = zip(v.inverse_transform(X)[0], X.A[0])

# sort results in ascending order
#results.sort(key=operator.itemgetter(1))

# sort results in descending order
results.sort(key=operator.itemgetter(1), reverse=True)

print results

with open('results/%s.tsv' % fname, 'wb') as f:
	f.write('\n'.join('%s %s' % x for x in results))