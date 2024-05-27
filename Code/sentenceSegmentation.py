from util import *

import re
import nltk.data


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# Punctuation used as sentence breakers:
		# Period .
		# Question mark ?
		# Exclamation mark !
		# Colon :
		# Ellipsis ...

		# Asserting that argument for function call is a string
		assert type(text) == str, f"Argument must be a string\nArgument given: {text}"

		# Sentence breakers used: ['.', '?', '!', ':', '...']
		segmentedText = [sentence for sentence in re.split(r'[.?!]+|\.{3}', text) if sentence]
		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
		assert type(text) == str, f"Argument must be a string\nArgument given: {text}"


		tokenizedSentences = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
		segmentedText = tokenizedSentences.tokenize(text)
		return segmentedText