# Add your import statements here
import math
import json
import string


# Add any utility functions here

def computeIDF(text):
    """
    Compute IDF for each word in the corpus.

    Parameters
    ----------
    text : list of lists of lists
        A list of documents where each document is represented as a list of sentences,
        and each sentence is represented as a list of tokens.

    Returns
    -------
    dict
        A dictionary where keys are unique words in the corpus and values
        are their IDF scores computed from the corpus.
    """
    totalDocuments = len(text)
    wordCounts = {}
        
    # Find count of docs containing the word
    for document in text:
        uniqueWordsInDoc = set(word for sentence in document for word in sentence)
        for word in uniqueWordsInDoc:
            wordCounts[word] = wordCounts.get(word, 0) + 1
    
    # Compute IDF
    idfs = {}
    for word, count in wordCounts.items():
        idfs[word] = math.log(totalDocuments / count)
    
    return idfs

def idfStopWords(filePath, outputFilePath, k = 100):
    """
    Compute the first k lowest IDF scores from a file containing a list of lists of lists.

    Parameters
    ----------
    filePath : str
        The file path containing the list of lists of lists.
    k : int, optional
        The number of words with the lowest IDF scores to return (default is 100).

    Returns
    -------
    list
        A list containing the first k words with the lowest IDF scores.
    """
    # Read data from the file
    with open(filePath, 'r') as file:
        data = json.load(file)  
    
    # Compute IDF scores
    idfs = computeIDF(data)
    
    # Sort IDF scores in ascending order and remove punctuations
    sorted_idfs = sorted(idfs.items(), key=lambda x: idfs.get(x[0], 0))
    sorted_idfs = [(word, idf) for word, idf in sorted_idfs if word not in string.punctuation]

    # Get the first k words with the lowest IDF scores
    stopWords = [word for word, idf in sorted_idfs[:k]]
    with open(outputFilePath, 'w') as f:
            json.dump(stopWords, f)    
    
    return stopWords

if __name__ == '__main__':
    print('\n', idfStopWords('output/reduced_docs.txt', 'idfStopwordsList.txt', k=179), '\n', sep='')
    
    

