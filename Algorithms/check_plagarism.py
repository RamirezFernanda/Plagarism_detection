'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''

import os  # For file management
import numpy as np  # For numerical computations
from joblib import Parallel, delayed  # For parallel processing
from nltk.tokenize import word_tokenize  # For natural language processing
from nltk.stem import WordNetLemmatizer, PorterStemmer  # For lemmatization and stemming
from sklearn.feature_extraction.text import TfidfVectorizer  # For machine learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
import nltk
nltk.download('punkt')
nltk.download('wordnet')

original_route = "./Detection_Files/Originals"
suspicious_route = "./Detection_Files/Suspicious"

original_files = os.listdir(original_route)
suspicious_files = os.listdir(suspicious_route)

def stemming(doc, stem=False):
    tokens = word_tokenize(doc.lower())
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

def compare_files(doc1, doc2, ngram, stem=True):

    doc1_stem = stemming(doc1, stem=stem)
    doc2_stem = stemming(doc2, stem=stem)
   
    vectorizer = TfidfVectorizer(ngram_range=(
        ngram, ngram), tokenizer=word_tokenize)  # Generate document vectors
    vectorize_doc1 = vectorizer.fit_transform([doc1_stem])
    vectorize_doc2 = vectorizer.transform([doc2_stem])
    return cosine_similarity(vectorize_doc1, vectorize_doc2)[0][0]



def read_documents():
    similarities = np.zeros((len(suspicious_files), len(original_files)))
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



def train_svm_model(similarities):
    # Creating the target vector
    target = np.zeros((len(suspicious_files), len(original_files)))
    for i in range(len(suspicious_files)):
        target[i, :] = (similarities[i, :] > 0.13).astype(int)
    
    # Creating the feature matrix
    features = similarities.reshape(-1, 1)
    
    # Training the SVM model
    clf = svm.SVC(kernel='linear')
    clf.fit(features, target.reshape(-1))
    
    return clf

def predict_plagiarism(clf, similarities):
    # Creating the feature matrix
    features = similarities.reshape(-1, 1)
    
    # Predicting the target vector
    target = clf.predict(features)
    target = target.reshape(len(suspicious_files), len(original_files))
    
    return target
