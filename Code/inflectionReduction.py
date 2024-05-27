from util import *
import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')

import spacy

class InflectionReduction:

	def reduce_stem(self, text):
		"""
		Stemming/Lemmatization
		
		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		stemmer = PorterStemmer()
		reducedText = []
		
		for sentence in text:
			# Perform stemming
			stemmedSentence = [stemmer.stem(word) for word in sentence]
			reducedText.append(stemmedSentence)
		
		return reducedText


	def reduce_lemma(self, text):
	    # Lemmatization
		# lemmatizer = nltk.WordNetLemmatizer()
		# reducedText = []

		# for sentence in text:
		# 	# Perform lemmatization
		# 	lemmatizedSentence = [lemmatizer.lemmatize(word) for word in sentence]
		# 	reducedText.append(lemmatizedSentence)

		# Alternate Approach using Stacy

		nlp = spacy.load('en_core_web_sm')
		reducedText = []
		for sentence in text:
			# Perform lemmatization
			lemmatizedSentence = [token.lemma_ for token in nlp(" ".join(sentence))]
			reducedText.append(lemmatizedSentence)

		return reducedText