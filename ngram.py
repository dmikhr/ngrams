# -*- coding: utf-8 -*-

# @dmitrypkh

from sys import argv
import operator

from sklearn.feature_extraction.text import CountVectorizer

HELP_MSG = \
"""
USAGE:

$ python ngram.py file_with_text number_of_words

"""

fname = argv[1]; N = int(argv[2])

stop_words = set([x.strip() for x in open('stopwords.en.txt', 'r')])

text = open('data/%s.txt' % fname, 'r').read()

# count n-gram frequencies
v = CountVectorizer(ngram_range=(N,N))
X = v.fit_transform([text])
ngramfreq = zip(v.inverse_transform(X)[0], X.A[0])

# remove n-gram if it contains only stop words (example: 'would even', 'are not', etc.)
remove_ngrams = []
for chunk, freq in ngramfreq:
	chunk_words = chunk.split(' ')
	if set(chunk_words).issubset(stop_words):
		#print chunk, freq
		remove_ngrams.append(tuple([chunk, freq]))
# removing
ngramfreq = list(set(ngramfreq) - set(remove_ngrams))

# sort ngramfreq in ascending order
#ngramfreq.sort(key=operator.itemgetter(1))

# sort ngramfreq in descending order
ngramfreq.sort(key=operator.itemgetter(1), reverse=True)
		
with open('results/%s.tsv' % fname, 'wb') as f:
	f.write('\n'.join('%s %s' % x for x in ngramfreq))
