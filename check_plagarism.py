# The necessary libraries are imported
import os  # For file management
import numpy as np  # For numerical computations
from joblib import Parallel, delayed  # For parallel processing
from nltk.tokenize import word_tokenize  # For natural language processing
from sklearn.feature_extraction.text import TfidfVectorizer  # For machine learning
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# Pre-trained sentence tokenizer to split text into individual sentences
nltk.download('punkt')

# Two directories are specified
# For the original documents
original_route = "./Detection_Files/Originals"
# For the suspicious documents
suspicious_route = "./Detection_Files/Suspicious"

# The list of original and suspicious files is obtained
original_files = os.listdir(original_route)
suspicious_files = os.listdir(suspicious_route)

# Function that takes two documents as input and returns their cosine similarity score
def compare_files(doc1, doc2, ngram_range):
  # Vectorizer extracts all possible 3-grams (sequences of 3 consecutive words) from the text
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(
        ngram_range, ngram_range), tokenizer=word_tokenize)  # Generate document vectors
    tfidf_doc1 = tfidf_vectorizer.fit_transform([doc1])
    tfidf_doc2 = tfidf_vectorizer.transform([doc2])
    return cosine_similarity(tfidf_doc1, tfidf_doc2)[0][0]


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

