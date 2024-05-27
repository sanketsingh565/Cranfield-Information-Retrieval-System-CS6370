from util import *
import re
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		
		# Punctuation used as word breakers:
		# Comma ,
		# Hyphen -
		# Semi colon ;
		# Apostrophe '
		# Quotations "
		# Parentheses ()
		# Curly brackets {}
		# Square Bracket []
		# Slash /
		

		# We are ensuring that input is in correct format in the sentenceSegmentation function
		# So, here we directly proceed
		tokenizedText = []
		# wordBreakers = r'[,\-;\'"\(\)\[\]\{\}]'
		wordBreakers = r'[,\-;\'"\(\)\[\]\{\}/]'

		for sentence in text:
			# ('\s+|') + wordBreakers specifies to tokenize at white spaces  
			# or at certain punctuations defined in wordBreakers
			tokens = re.split(r'\s+|' + wordBreakers, sentence)	

			tokens = [token.strip() for token in tokens if token.strip()]
			tokenizedText.append(tokens)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		
		tokenizedText = [TreebankWordTokenizer().tokenize(sentence, convert_parentheses=True) for sentence in text]

		return tokenizedText