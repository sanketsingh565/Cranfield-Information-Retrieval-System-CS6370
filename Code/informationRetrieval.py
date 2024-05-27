from util import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
import time
# Add your import statements here



class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.termDocMat = None
		self.idf_mat = None
		self.term_position_map = None
		self.doc_ids = None
		self.doc2vec_model = None
		self.corpus_size = None
		self.word2vec_model = None
		self.doc_embeddings = None
		self.bm25 = None
		self.IDs = None

	def buildIndex(self, docs, docIDs, 
					doc2vec=False, doc2vec_dims=400, doc2vec_lr=0.135, doc2vec_epochs=500,
					word2vec=False, word2vec_dims=400, word2vec_lr=0.03, word2vec_epochs=500,
					bm25=False, k1=1.5, b=.75
					):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		self.corpus_size = len(docIDs)
		self.IDs = docIDs

		if doc2vec:
			print(f'doc2vec_dims={doc2vec_dims}, doc2vec_lr={doc2vec_lr}, doc2vec_epochs={doc2vec_epochs}')
			self.doc2vec_model = self.Doc2Vec_train(docs, doc2vec_dims, doc2vec_lr, doc2vec_epochs)
			return
		
		if word2vec:
			print(f'word2vec_dims={word2vec_dims}, word2vec_lr={word2vec_lr}, word2vec_epochs={word2vec_epochs}')
			self.word2vec_model = self.Word2Vec_train(docs, word2vec_dims, word2vec_lr, word2vec_epochs)
			self.doc_embeddings = self.generate_doc_embedding(self.word2vec_model, docs, word2vec_dims)
			return
		
		if bm25:
			print('METHOD-BM25')
			tokenized_docs = [[word for sublist in sublist_list for word in sublist] for sublist_list in docs]
			self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
			return
		
		index = {}
		self.term_doc_matrix = {}

		# Compute term frequencies
		for doc_index, doc in enumerate(docs):
			tokenized_doc = [term for sentence in doc for term in sentence]

			for term in set(tokenized_doc):
				# Compute term frequency for a document
				tf = np.count_nonzero(np.array(tokenized_doc) == term)
				if term in index.keys():
					index[term].append([docIDs[doc_index], tf])	
				else:
					index[term] = [[docIDs[doc_index], tf]]
		
		self.index = index
		
		self.termDocumentMatrix(docs, docIDs)

	
	def Word2Vec_train(self, docs, word2vec_dims, word2vec_lr, word2vec_epochs):
		tokenized_docs = [[word for sublist in sublist_list for word in sublist] for sublist_list in docs]
		
		word2vec_model = Word2Vec(sentences=tokenized_docs, 
								  vector_size=word2vec_dims, 
								  window=2, 
								  workers=4,
								  sg=1, 
								 )
		

		word2vec_model.train(tokenized_docs, total_examples=len(tokenized_docs), epochs=word2vec_epochs, start_alpha=word2vec_lr)
		return word2vec_model


	def generate_doc_embedding(self, model, docs, dims):
		tokenized_docs = [[word for sublist in sublist_list for word in sublist] for sublist_list in docs]
		doc_embeddings = []
		
		for doc in tokenized_docs:
			doc_embed = np.zeros(dims)
			words_in_doc = 0
			for term in doc:
				# Skip new query words
				if term not in model.wv.index_to_key:
					continue	

				words_in_doc += 1
				doc_embed += model.wv[term]
			
			if words_in_doc != 0:
				doc_embed /= words_in_doc
			doc_embeddings.append(doc_embed)
			
		return doc_embeddings


	def Doc2Vec_train(self, docs, doc2vec_dims, doc2vec_lr, doc2vec_epochs):
		# Settings tags to each document
		tokenized_docs = [[word for sublist in sublist_list for word in sublist] for sublist_list in docs]
		taggedDocuments = [TaggedDocument(words=row, tags=[str(i)]) for i, row in enumerate(tokenized_docs)]

		model = Doc2Vec(vector_size=doc2vec_dims,
						alpha=doc2vec_lr,
						min_count=1,
						window=2,
						dm=0,
						min_alpha=0.0025,
						#shrink_windows=True,
						epochs = doc2vec_epochs)  # 0 = DBOW; 1 = DM

		model.build_vocab(taggedDocuments)
		model.train(taggedDocuments,
					total_examples = model.corpus_count,
					epochs = model.epochs)
		return model



	def termDocumentMatrix(self, docs, docIDs):
		"""
		Document this properly
		"""
		mat = self.index
		# Number of documents in corpus
		N = len(docs)
		idf_mat = {}
		# Compute tf-idf
		for term in mat.keys():
			n = len(mat[term])
			idf_term = np.log10(N/n)
			idf_mat[term] = idf_term
			mat[term] = [[doc_id, tfidf * idf_term] for doc_id, tfidf in mat[term]]

		term_position_map = {term:position for position, term in enumerate(self.index.keys())}

		termDocMat = {}
		vocab_size = len(self.index.keys())

		for doc_id in docIDs:
			termDocMat[doc_id] = [0]*vocab_size
		
		for term in self.index.keys():	
			for doc_id, tf_idf in mat[term]:			
				termDocMat[doc_id][term_position_map[term]] = tf_idf
				
		self.termDocMat = termDocMat
		self.idf_mat = idf_mat
		self.term_position_map = term_position_map


	def rank(self, queries, LSA=False, LSA_dims = 400, doc2vec=False, word2vec=False, bm25=False):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		
		if bm25:
			bm25_doc_IDs_ordered = self.rank_bm25(queries)
			file_path = 'bm25_retrievals.txt'

			# Open the file in write mode
			with open(file_path, 'w') as file:
				for item in bm25_doc_IDs_ordered:
					file.write(str(item) + '\n')

			return bm25_doc_IDs_ordered		


		if word2vec:
			word2vec_doc_IDs_ordered = self.rank_word2vec(queries)
			file_path = 'word2vec_retrievals.txt'

			# Open the file in write mode
			with open(file_path, 'w') as file:
				for item in word2vec_doc_IDs_ordered:
					file.write(str(item) + '\n')

			return word2vec_doc_IDs_ordered

		if doc2vec:
			doc2vec_doc_IDs_ordered = self.rank_doc2vec(queries)
			file_path = 'doc2vec_retrievals.txt'

			# Open the file in write mode
			with open(file_path, 'w') as file:
				for item in doc2vec_doc_IDs_ordered:
					file.write(str(item) + '\n')

			return doc2vec_doc_IDs_ordered		



		# VSM approach
		doc_IDs_ordered = []

		start_time = time.time()

		for query in queries:
			# Flattening queries
			tokenized_query = [term for sentence in query for term in sentence]			
			query_vector = [0]*(len(self.index.keys()))

			# Compute tf-idf vector for query
			for term in set(tokenized_query):
				tf = np.count_nonzero(np.array(tokenized_query) == term)
				if term not in self.idf_mat.keys():
					continue
				query_vector[self.term_position_map[term]] = tf * self.idf_mat[term]

			doc_vectors = list(self.termDocMat.values())

			# Compute cosine similarity
			cosine_similarities = cosine_similarity([query_vector], doc_vectors)

			
			doc_ids = list(self.termDocMat.keys())
			self.doc_ids = doc_ids
			
			doc_similarity_pairs = zip(doc_ids, cosine_similarities[0])

			# Rank in decreasing order of similarity
			similar_documents = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)
			similar_documents = [int(doc_id) for doc_id, _ in similar_documents]
			doc_IDs_ordered.append(similar_documents)

		end_time = time.time()	
		runtime = end_time - start_time
		

		# LSA approach
		if LSA:
			LSA_doc_IDs_ordered =  self.rank_LSA(queries, LSA_dims)
			# Specify the file path
			file_path = 'LSA_retrievals.txt'


			# Open the file in write mode
			with open(file_path, 'w') as file:
				# Iterate over each element in the list
				for item in LSA_doc_IDs_ordered:
					# Write the element to the file followed by a newline character
					file.write(str(item) + '\n')
			
			return LSA_doc_IDs_ordered	

		print("VSM runtime: ", runtime)

		# Specify the file path
		file_path = 'retrievals.txt'
		# Open the file in write mode
		with open(file_path, 'w') as file:
			# Iterate over each element in the list
			for item in doc_IDs_ordered:
				# Write the element to the file followed by a newline character
				file.write(str(item) + '\n')

		# Return VSM retieval
		return doc_IDs_ordered
	

	def rank_bm25(self, queries):
		bm25_doc_IDs_ordered = []
		
		for query in queries:
			tokenized_query = [term for sentence in query for term in sentence]		
			# similarities = self.bm25.get_scores(tokenized_query)
			most_sim_docs_scores = self.bm25.get_scores(tokenized_query)
						
			# Sorting based on scores
			sorted_doc_query_sims = np.argsort(most_sim_docs_scores)[::-1] 

			# Reverse hashing to retrieve document IDs
			sorted_doc_query_sims = [index + 1 for index in sorted_doc_query_sims]		
			
			bm25_doc_IDs_ordered.append(sorted_doc_query_sims)

		return bm25_doc_IDs_ordered


	def rank_word2vec(self, queries):		
		word2vec_doc_IDs_ordered = []
		
		dims = len(self.doc_embeddings[0])

		query_embeddings = self.generate_doc_embedding(self.word2vec_model, queries, dims)

		for emb in query_embeddings:
			# Computing cosine similarity
			similarities = [cosine_similarity([emb], [doc_vector])[0][0] for doc_vector in self.doc_embeddings]

			ranked_args = np.argsort(similarities)[::-1]

			# Reverse hashing
			sorted_doc_query_sims = [index + 1 for index in ranked_args]
			word2vec_doc_IDs_ordered.append(sorted_doc_query_sims)
	
		return word2vec_doc_IDs_ordered
		

	def rank_doc2vec(self, queries):
		model = self.doc2vec_model
		
		doc2vec_doc_IDs_ordered = []
		
		for query in queries:
			tokenized_query = [term for sentence in query for term in sentence]	
			query_vector = model.infer_vector(tokenized_query)
			similar_docs_scores = model.dv.most_similar([query_vector], topn=self.corpus_size)
			similar_docs = [int(item[0])+1 for item in similar_docs_scores]
			doc2vec_doc_IDs_ordered.append(similar_docs)
	
		return doc2vec_doc_IDs_ordered
	

	def rank_LSA(self, queries, LSA_dims):

		terms = list(self.index.keys())
		# Term frequency - doc matrix (X)
		X = np.zeros((len(self.index.keys()), len(self.doc_ids)))

		# Construct X matrix
		for term_index, term in enumerate(terms):
			for docId_tf in self.index[term]:
				# Hashing documents
				X[term_index][int(docId_tf[0])-1] = docId_tf[1]

		# Singular Value Decomposition (SVD)
		T0, S0, D0_T = np.linalg.svd(X)

		# Reduce SVD dimensions
		D_T = D0_T[:LSA_dims]
		T = T0[:, :LSA_dims]
		S = np.diag(S0[:LSA_dims])

		LSA_doc_IDs_ordered = []
	
		for query in queries:
			# Tokenize the query
			tokenized_query = [term for sentence in query for term in sentence]	
			X_q_t = np.zeros((1, len(self.index.keys())))

			# Computing X_q_t
			for term in set(tokenized_query):
				tf = np.count_nonzero(np.array(tokenized_query) == term)
				if term not in terms:
					continue
				X_q_t[0][terms.index(term)] = tf

			S_inv = np.linalg.inv(S)

			# Query vector
			D_q = np.matmul(X_q_t, np.matmul(T, S_inv))

			D = np.transpose(D_T)

			# Reshape q into a 2D array to compute CSR
			q = D_q.reshape(1, -1)
			doc_query_sims = cosine_similarity(D, q).flatten()

			# Sorting based on CSR
			sorted_doc_query_sims = np.argsort(doc_query_sims)[::-1] 

			# Reverse hashing to retrieve document IDs
			sorted_doc_query_sims = [index + 1 for index in sorted_doc_query_sims]
			
			LSA_doc_IDs_ordered.append(sorted_doc_query_sims)

		return LSA_doc_IDs_ordered
	

	# def storeRetrievals(filePath):
