from util import *
from nltk.corpus import stopwords


class StopwordRemoval():

	
	def fromIDF(self, text):
		"""
		Remove stopwords from a list of tokenized sentences using a custom list of stopwords
		loaded using IDF approach.

		Parameters
		----------
		text : list
			A list of lists where each sub-list represents a sequence of tokens
			representing  a sentence.

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		# Bottom-up approach for stopword
		with open('idfStopwordsList.txt', 'r') as file:
			stopWords_self = json.load(file)


		# Remove stopwords from text
		for sentence in text:
			# Remove stopwords from each list
			filteredWords = [word for word in sentence if word.lower() not in stopWords_self]
			stopwordRemovedText.append(filteredWords)

		return stopwordRemovedText
		

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		stopWords_nltk = stopwords.words('english')
		stopwordRemovedText = []

		# Remove stopwords from text
		for sentence in text:
			# Remove stopwords from each list
			filteredWords = [word for word in sentence if word.lower() not in stopWords_nltk]
			stopwordRemovedText.append(filteredWords)
			
		return stopwordRemovedText



	