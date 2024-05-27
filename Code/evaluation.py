# from util import *
# import math

# # Add your import statements here


# class Evaluation():

#     def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
#         """
# 		Computation of precision of the Information Retrieval System
# 		at a given value of k for a single query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of integers denoting the IDs of documents in
# 			their predicted order of relevance to a query
# 		arg2 : int
# 			The ID of the query in question
# 		arg3 : list
# 			The list of IDs of documents relevant to the query (ground truth)
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The precision value as a number between 0 and 1
#         """

#         relevant_docs = set(true_doc_IDs)
#         retrieved_docs = query_doc_IDs_ordered[:k]
#         num_relevant_retrieved = len(relevant_docs.intersection(retrieved_docs))
#         precision = num_relevant_retrieved / k if k > 0 else 0
#         return precision

#     def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
#         """
# 		Computation of precision of the Information Retrieval System
# 		at a given value of k, averaged over all the queries

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		arg2 : list
# 			A list of IDs of the queries for which the documents are ordered
# 		arg3 : list
# 			A list of dictionaries containing document-relevance
# 			judgements - Refer cran_qrels.json for the structure of each
# 			dictionary
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The mean precision value as a number between 0 and 1
#         """

#         # total_precision = sum(self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrels[i][docs], k) for i in range(len(query_ids)))
#         # meanPrecision = total_precision / len(query_ids) if len(query_ids) > 0 else 0
#         # return meanPrecision

#         precisions = []
#         for i in range(len(query_ids)):
#             query = query_ids[i]
            

#     def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
#         """
# 		Computation of recall of the Information Retrieval System
# 		at a given value of k for a single query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of integers denoting the IDs of documents in
# 			their predicted order of relevance to a query
# 		arg2 : int
# 			The ID of the query in question
# 		arg3 : list
# 			The list of IDs of documents relevant to the query (ground truth)
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The recall value as a number between 0 and 1        
#         """

#         relevant_docs = set(true_doc_IDs)
#         retrieved_docs = query_doc_IDs_ordered[:k]
#         num_relevant_retrieved = len(relevant_docs.intersection(retrieved_docs))
#         recall = num_relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
#         return recall

#     def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
#         """
# 		Computation of recall of the Information Retrieval System
# 		at a given value of k, averaged over all the queries

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		arg2 : list
# 			A list of IDs of the queries for which the documents are ordered
# 		arg3 : list
# 			A list of dictionaries containing document-relevance
# 			judgements - Refer cran_qrels.json for the structure of each
# 			dictionary
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The mean recall value as a number between 0 and 1
#         """

#         total_recall = sum(self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrels[i][docs], k) for i in range(len(query_ids)))
#         meanRecall = total_recall / len(query_ids) if len(query_ids) > 0 else 0
#         return meanRecall

#     def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
#         """
# 		Computation of fscore of the Information Retrieval System
# 		at a given value of k for a single query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of integers denoting the IDs of documents in
# 			their predicted order of relevance to a query
# 		arg2 : int
# 			The ID of the query in question
# 		arg3 : list
# 			The list of IDs of documents relevant to the query (ground truth)
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The fscore value as a number between 0 and 1
#         """

#         precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
#         recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
#         fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         return fscore

#     def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
#         """
# 		Computation of fscore of the Information Retrieval System
# 		at a given value of k, averaged over all the queries

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		arg2 : list
# 			A list of IDs of the queries for which the documents are ordered
# 		arg3 : list
# 			A list of dictionaries containing document-relevance
# 			judgements - Refer cran_qrels.json for the structure of each
# 			dictionary
# 		arg4 : int
# 			The k value
		
# 		Returns
# 		-------
# 		float
# 			The mean fscore value as a number between 0 and 1
#         """

#         total_fscore = sum(self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrels[i][docs], k) for i in range(len(query_ids)))
#         meanFscore = total_fscore / len(query_ids) if len(query_ids) > 0 else 0
#         return meanFscore

#     def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
#         """
# 		Computation of nDCG of the Information Retrieval System
# 		at given value of k for a single query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of integers denoting the IDs of documents in
# 			their predicted order of relevance to a query
# 		arg2 : int
# 			The ID of the query in question
# 		arg3 : list
# 			The list of IDs of documents relevant to the query (ground truth)
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The nDCG value as a number between 0 and 1        
#         """

#         DCG = 0
#         relevant_docs = set(true_doc_IDs)
#         for i in range(k):
#             doc_id = query_doc_IDs_ordered[i]
#             if doc_id in relevant_docs:
#                 DCG += 1 / math.log2(i + 2)  # i+2 because index starts from 0
#         IDCG = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_docs))))
#         nDCG = DCG / IDCG if IDCG > 0 else 0
#         return nDCG

#     def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
#         """
# 		Computation of nDCG of the Information Retrieval System
# 		at a given value of k, averaged over all the queries

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		arg2 : list
# 			A list of IDs of the queries for which the documents are ordered
# 		arg3 : list
# 			A list of dictionaries containing document-relevance
# 			judgements - Refer cran_qrels.json for the structure of each
# 			dictionary
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The mean nDCG value as a number between 0 and 1        
#         """

#         total_ndcg = sum(self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrels[i][docs], k) for i in range(len(query_ids)))
#         meanNDCG = total_ndcg / len(query_ids) if len(query_ids) > 0 else 0
#         return meanNDCG

#     def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
#         """
# 		Computation of average precision of the Information Retrieval System
# 		at a given value of k for a single query (the average of precision@i
# 		values for i such that the ith document is truly relevant)

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of integers denoting the IDs of documents in
# 			their predicted order of relevance to a query
# 		arg2 : int
# 			The ID of the query in question
# 		arg3 : list
# 			The list of documents relevant to the query (ground truth)
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The average precision value as a number between 0 and 1        
#         """

#         relevant_docs = set(true_doc_IDs)
#         num_relevant_docs = len(relevant_docs)
#         cumulative_precision = 0
#         num_relevant_retrieved = 0
#         for i in range(k):
#             doc_id = query_doc_IDs_ordered[i]
#             if doc_id in relevant_docs:
#                 num_relevant_retrieved += 1
#                 cumulative_precision += num_relevant_retrieved / (i + 1)
#         avgPrecision = cumulative_precision / num_relevant_docs if num_relevant_docs > 0 else 0
#         return avgPrecision

#     def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
#         """
# 		Computation of MAP of the Information Retrieval System
# 		at given value of k, averaged over all the queries

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		arg2 : list
# 			A list of IDs of the queries
# 		arg3 : list
# 			A list of dictionaries containing document-relevance
# 			judgements - Refer cran_qrels.json for the structure of each
# 			dictionary
# 		arg4 : int
# 			The k value

# 		Returns
# 		-------
# 		float
# 			The MAP value as a number between 0 and 1
#         """

#         total_avg_precision = sum(self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], qrels[i][docs], k) for i in range(len(query_ids)))
#         meanAveragePrecision = total_avg_precision / len(query_ids) if len(query_ids) > 0 else 0
#         return meanAveragePrecision


import math

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, qrels, k):
        relevant_docs = [int(entry['id']) for entry in qrels if int(entry['query_num']) == int(query_id)]
        retrieved_docs = query_doc_IDs_ordered[:k]
        num_relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_docs)))
        precision = num_relevant_retrieved / k if k > 0 else 0
        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
        """

        total_precision = sum(self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrels, k) for i in range(len(query_ids)))
        meanPrecision = total_precision / len(query_ids) if len(query_ids) > 0 else 0
        return meanPrecision


    def queryRecall(self, query_doc_IDs_ordered, query_id, qrels, k):
        relevant_docs = [int(entry['id']) for entry in qrels if int(entry['query_num']) == int(query_id)]
        retrieved_docs = query_doc_IDs_ordered[:k]
        num_relevant_retrieved = len(set(relevant_docs).intersection(retrieved_docs))
        recall = num_relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
        """

        total_recall = sum(self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrels, k) for i in range(len(query_ids)))
        meanRecall = total_recall / len(query_ids) if len(query_ids) > 0 else 0
        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, qrels, k):
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, qrels, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, qrels, k)
        fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
        """

        total_fscore = sum(self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrels, k) for i in range(len(query_ids)))
        meanFscore = total_fscore / len(query_ids) if len(query_ids) > 0 else 0
        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
        relevant_docs = {int(entry['id']): 5 - int(entry['position']) for entry in qrels if int(entry['query_num']) == int(query_id)}
                
        DCG = 0
        
        relevant_scores= [] # Dictionary with keys as doc_id and value as relevance score
        for i in range(k):
            doc_id = int(query_doc_IDs_ordered[i])
            if doc_id in relevant_docs.keys():
                DCG += relevant_docs[doc_id]/math.log2(i+2)
                relevant_scores.append(relevant_docs[doc_id])
        
        ideal_relevance = [value for key, value in sorted(relevant_docs.items(), key=lambda item: item[1], reverse=True)[:k]]		

        # ideal_relevance = sorted(relevant_scores, reverse=True)
        IDCG = sum(ideal_relevance[i] / math.log2(idx + 2) for idx, i in enumerate(range(len(ideal_relevance))))
        
        nDCG = DCG / IDCG if IDCG > 0 else 0
        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1        
        """
        
        total_ndcg = sum(self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrels, k) for i in range(len(query_ids)))
        meanNDCG = total_ndcg / len(query_ids) if len(query_ids) > 0 else 0
        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, qrels, k):
        relevant_docs = {int(entry['id']) for entry in qrels if int(entry['query_num']) == int(query_id)}
        num_relevant_docs = len(relevant_docs)
        cumulative_precision = 0
        num_relevant_retrieved = 0
        
        for i in range(k):
            doc_id = query_doc_IDs_ordered[i]
            if doc_id in relevant_docs:
                num_relevant_retrieved += 1
                cumulative_precision += num_relevant_retrieved / (i + 1)
                
        avgPrecision = cumulative_precision / num_relevant_docs if num_relevant_docs > 0 else 0
        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
        """

        total_avg_precision = sum(self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], qrels, k) for i in range(len(query_ids)))
        meanAveragePrecision = total_avg_precision / len(query_ids) if len(query_ids) > 0 else 0
        return meanAveragePrecision