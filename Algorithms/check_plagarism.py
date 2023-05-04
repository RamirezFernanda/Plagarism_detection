'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''

# The necessary libraries are imported
import os  # For file management
import numpy as np  # For numerical computations
from joblib import Parallel, delayed  # For parallel processing
from nltk.tokenize import word_tokenize  # For natural language processing
from nltk.stem import WordNetLemmatizer, PorterStemmer  # For lemmatization and stemming
from sklearn.feature_extraction.text import TfidfVectorizer  # For machine learning
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# Pre-trained sentence tokenizer to split text into individual sentences
nltk.download('punkt')
nltk.download('wordnet')

# Two directories are specified
# For the original documents
original_route = "./Detection_Files/Originals"

# For the suspicious documents
suspicious_route = "./Detection_Files/Suspicious"
# The list of original and suspicious files is obtained
original_files = os.listdir(original_route)
suspicious_files = os.listdir(suspicious_route)

# Function that takes two documents as input and returns their cosine similarity score

def stemming(doc, stem=False):
    tokens = word_tokenize(doc.lower())
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

def compare_files(doc1, doc2, ngram, stem=True):
#  Preprocess documents by tokenizing and stemming their words, and joining them back into strings
    doc1_stem = stemming(doc1, stem=stem)
    doc2_stem = stemming(doc2, stem=stem)
    # Vectorizer extracts all possible n-grams (sequences of n consecutive words) from the text
    vectorizer = TfidfVectorizer(ngram_range=(
        ngram, ngram), tokenizer=word_tokenize)  # Generate document vectors
    vectorize_doc1 = vectorizer.fit_transform([doc1_stem])
    vectorize_doc2 = vectorizer.transform([doc2_stem])
    return cosine_similarity(vectorize_doc1, vectorize_doc2)[0][0]


def read_documents():
    # Performs pairwise document comparison between all suspicious and original documents and generates
    #  a similarity matrix containing the similarity scores between each pair of documents
    similarities = np.zeros((len(suspicious_files), len(original_files)))
    # The first loop iterates over each suspicious document, and the second loop iterates over each original document
    # For each pair of documents, the text content of each document is read in and passed to the  function to calculate the similarity score
    for i, plagiarized_file in enumerate(suspicious_files):
        sus_route = os.path.join(suspicious_route, plagiarized_file)
        with open(sus_route, 'r', encoding="utf-8") as plagiarized_file:
            plagiarized_text = plagiarized_file.read()
        for j, original_file in enumerate(original_files):
            org_route = os.path.join(original_route, original_file)
            with open(org_route, 'r', encoding="utf-8") as original_file:
                original_text = original_file.read()
                # The similarity score is then stored in the array at the corresponding location
            similarities[i, j] = compare_files(
                original_text, plagiarized_text, 3)
    return similarities
