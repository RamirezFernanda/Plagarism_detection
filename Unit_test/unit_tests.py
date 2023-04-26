'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''

from sklearn.metrics.pairwise import cosine_similarity
import os  # For file management
import numpy as np  # For numerical computations
from joblib import Parallel, delayed  # For parallel processing
from nltk.tokenize import word_tokenize  # For natural language processing
from sklearn.feature_extraction.text import TfidfVectorizer  # For machine learning
import nltk

# Two directories are specified
# For the original documents
original_route = "./Detection_Files/Originals"
# For the suspicious documents
suspicious_route = "./Detection_Files/Suspicious"

# The list of original and suspicious files is obtained
original_files = os.listdir(original_route)
suspicious_files = os.listdir(suspicious_route)

print(
    f'Original files found in the path {original_route}: {original_files} \n -------------- \nTotal files: {len(original_files)}\n --------------')
print(
    f'Suspicious files found in the path {suspicious_route}: {suspicious_files} \n -------------- \nTotal files: {len(suspicious_files)}\n --------------')


def compare_files(doc1, doc2, ngram_range):
  # Uses N-grams to extract features from the text data
  # Vectorizer will extract all possible 3-grams (sequences of 3 consecutive words) from the text
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(
        ngram_range, ngram_range), tokenizer=word_tokenize)  # Generate document vectors
    tfidf_doc1 = tfidf_vectorizer.fit_transform([doc1])
    tfidf_doc2 = tfidf_vectorizer.transform([doc2])
    return cosine_similarity(tfidf_doc1, tfidf_doc2)[0][0]


def unit_test_compare_files(doc1, doc2, ngram):
    similarity_score = compare_files(doc1, doc2, ngram)
    return similarity_score


print(f'La similitud para la prueba unitaria es:\n{unit_test_compare_files("/content/drive/MyDrive/Copia de construccion/documentos-genuinos/org-001.txt","/content/drive/MyDrive/Copia de construccion/documentos-con texto de otros/plg-001.txt",1)}')
