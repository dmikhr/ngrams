# -*- coding: utf-8 -*-

# @dmitrypkh

from sys import argv
import operator
import md5
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import CountVectorizer

HELP_MSG = \
"""
USAGE:

$ python ngram.py file_with_text number_of_words

"""

def processChunk(chunk):
	# count n-gram frequencies
	v = CountVectorizer(ngram_range=(N,N))
	X = v.fit_transform([chunk])
	ngramfreq = zip(v.inverse_transform(X)[0], X.A[0])

	return ngramfreq

def rm_stopwords(ngramfreq, stop_words):
	# remove n-gram if it contains only stop words (example: 'would even', 'are not', etc.)
	remove_ngrams = []
	for chunk, freq in ngramfreq:
		chunk_words = chunk.split(' ')
		if set(chunk_words).issubset(stop_words):
			#print chunk, freq
			remove_ngrams.append(tuple([chunk, freq]))
	# removing
	ngramfreq = list(set(ngramfreq) - set(remove_ngrams))
	
	return ngramfreq

def text_to_sentences(text):
	# make each sentence ends with dot
	text = text.replace('!', '.').replace('?', '.')
	# remove commas, semicolons and parenthesis
	text = text.replace(',','').replace(';','')
	text = text.replace('[','').replace(']','')
	text = text.replace('(','').replace(')','')
	# remove multiple spaces
	text = ' '.join(text.split())
	sentences = text.split('.')
	# strip spaces and new line characters from start and end of each string
	sentences = map(str.strip, sentences)
	# remove empty strings
	sentences = filter(None, sentences)
	return sentences

# return list of n-grams with their frequencies
def get_ngram_freq(sentences):
	# initialize dictionaries for n-gram frequencies
	ngrams_freq = {}
	freq_dict = {}
	for sentence in sentences:
		try:
			# get n-gram frequencies
			ngrams_freq_raw = processChunk(sentence)
			# remove n-grams that contain only stop words
			ngrams_freq_raw = rm_stopwords(ngrams_freq_raw, stop_words)
			# convert to dictionary
			ngrams_freq = dict(ngrams_freq_raw)
		except:
			pass
		for ngram, freq in ngrams_freq.iteritems():
			if ngram in freq_dict:
				freq_dict[ngram] += freq
			else:
				freq_dict[ngram] = freq

	freq_list = freq_dict.items()
	return freq_list

# get parameters from command line
fname = argv[1]; N = int(argv[2])

# load stop words
stop_words = set([x.strip() for x in open('stopwords.en.txt', 'r')])

# read file with text by line
text_lines = open('data/%s.txt' % fname, 'r').read().splitlines()

# join strings
text = ' '.join(text_lines)

# text to sentences, clear data
sentences = text_to_sentences(text)

# get n-grams with their frequencies
freq_list = get_ngram_freq(sentences)

# sort ngramfreq in ascending order
#freq_list.sort(key=operator.itemgetter(1))

# sort ngramfreq in descending order
freq_list.sort(key=operator.itemgetter(1), reverse=True)

# save results
with open('results/%s.tsv' % fname, 'wb') as f:
	f.write('\n'.join('%s %s' % x for x in freq_list).encode('utf8'))
