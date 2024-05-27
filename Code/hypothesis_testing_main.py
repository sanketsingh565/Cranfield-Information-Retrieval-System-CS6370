import json
import argparse
from sys import version_info

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

class SearchEngine:

    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        if self.args.reducer == "stemming":
            return self.inflectionReducer.reduce_stem(text)
        elif self.args.reducer == "lemmatization":
            return self.inflectionReducer.reduce_lemma(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs	
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP 
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        precisions_at_1, recalls_at_1, fscores_at_1, MAPs_at_1, nDCGs_at_1 = [], [], [], [], []
        precisions_at_10, recalls_at_10, fscores_at_10, MAPs_at_10, nDCGs_at_10 = [], [], [], [], []
        evaluation_results = {}
        # Read queries
        for i in range(1, 26):
            queries_json = json.load(open(args.dataset + f"/cran_queries_{i}.json", 'r'))[:]
            query_ids, queries = [item["query number"] for item in queries_json], \
                [item["query"] for item in queries_json]
            # Process queries 
            processedQueries = self.preprocessQueries(queries)

            # Read documents
            docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
            doc_ids, docs = [item["id"] for item in docs_json], \
                [item["body"] + item["title"] for item in docs_json]
            # [item["body"] for item in docs_json]
            # Process documents
            processedDocs = self.preprocessDocs(docs)

            # Build document index
            self.informationRetriever.buildIndex(processedDocs, doc_ids, doc2vec=args.doc2vec, word2vec=args.word2vec, bm25=args.bm25)
            # Rank the documents for each query
            doc_IDs_ordered = self.informationRetriever.rank(processedQueries, LSA=args.lsa, doc2vec=args.doc2vec, word2vec=args.word2vec, bm25=args.bm25)

            # Read relevance judgements
            qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

            # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
            precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
            for k in range(1, 11):
                precision = self.evaluator.meanPrecision(
                    doc_IDs_ordered, query_ids, qrels, k)
                precisions.append(precision)
                recall = self.evaluator.meanRecall(
                    doc_IDs_ordered, query_ids, qrels, k)
                recalls.append(recall)
                fscore = self.evaluator.meanFscore(
                    doc_IDs_ordered, query_ids, qrels, k)
                fscores.append(fscore)
                print("Precision, Recall and F-score @ " +
                      str(k) + " : " + str(precision) + ", " + str(recall) +
                      ", " + str(fscore))
                MAP = self.evaluator.meanAveragePrecision(
                    doc_IDs_ordered, query_ids, qrels, k)
                MAPs.append(MAP)
                nDCG = self.evaluator.meanNDCG(
                    doc_IDs_ordered, query_ids, qrels, k)
                nDCGs.append(nDCG)

            precisions_at_1.append(precisions[0])
            recalls_at_1.append(recalls[0])
            fscores_at_1.append(fscores[0])
            MAPs_at_1.append(MAPs[0])
            nDCGs_at_1.append(nDCGs[0])
            precisions_at_10.append(precisions[9])
            recalls_at_10.append(recalls[9])
            fscores_at_10.append(fscores[9])
            MAPs_at_10.append(MAPs[9])
            nDCGs_at_10.append(nDCGs[9])

        
        evaluation_results["precision_at_1"] = precisions_at_1
        evaluation_results["precisions_at_10"] = precisions_at_10
        evaluation_results["recalls_at_1"] = recalls_at_1
        evaluation_results["recalls_at_10"] = recalls_at_10
        evaluation_results["fscores_at_1"] = fscores_at_1
        evaluation_results["fscores_at_10"] = fscores_at_10
        evaluation_results["MAPs_at_1"] = MAPs_at_1
        evaluation_results["MAPs_at_10"] = MAPs_at_10
        evaluation_results["nDCGs_at_1"] = nDCGs_at_1
        evaluation_results["nDCGs_at_10"] = nDCGs_at_10

        

        if args.doc2vec:
            file_path = "doc2vec_metrics.json"
            # Save the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(evaluation_results, json_file)
        
        elif args.word2vec:
            file_path = "word2vec_metrics.json"
            # Save the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(evaluation_results, json_file)

        elif args.bm25:
            file_path = "bm25_metrics.json"
            # Save the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(evaluation_results, json_file)

        elif args.lsa:
            file_path = "lsa_metrics.json"
            # Save the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(evaluation_results, json_file)
        else:
            file_path = "vsm_metrics.json"
            # Save the dictionary to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(evaluation_results, json_file)

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
            [item["body"] + item["title"] for item in docs_json]
        # [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids, doc2vec=args.doc2vec, word2vec=args.word2vec, bm25=args.bm25)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery], LSA=args.lsa, doc2vec=args.doc2vec, word2vec=args.word2vec, bm25=args.bm25)[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default="cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="output/",
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-reducer', default="stemming",
                        help="Sentence Segmenter Type [stemming|lemmatization]")
    parser.add_argument('-tokenizer', default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")
    parser.add_argument('-lsa', action="store_true",
                        help="LSA set to store_true")
    parser.add_argument('-doc2vec', action="store_true",
                        help="doc2vec set to store_true")
    parser.add_argument('-word2vec', action="store_true",
                        help="word2vec set to store_true")
    parser.add_argument('-bm25', action="store_true",
                        help="bm25 set to store_true")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    # Either handle query from user or evaluate on the complete dataset 
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()

   
