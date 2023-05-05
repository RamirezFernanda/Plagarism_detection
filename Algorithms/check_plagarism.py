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
from sklearn.metrics.pairwise import cosine_similarity # For Cosine Similarity
from sklearn import svm # Support vector machine (SVM) algorithm from the scikit-learn library
import nltk # Natural Language Toolkit (NLTK)  for processing and analyzing text data

#Pre-trained sentence tokenizer to split text into individual sentences
nltk.download('punkt')
# Downloads the WordNet corpus from the Natural Language Toolkit (nltk) library
nltk.download('wordnet')

#Two directories are specified
#For the original documents
original_route = "./Detection_Files/Originals"
#For the suspicious documents
suspicious_route = "./Detection_Files/Suspicious"

#The list of original and suspicious files is obtained
original_files = os.listdir(original_route)
suspicious_files = os.listdir(suspicious_route)

# Function that takes a document (a string) and an optional boolean parameter "stem"
#If stem is True applies stemming to each token in the tokenized document.
def stemming(doc, stem=False):
    tokens = word_tokenize(doc.lower()) # Tokenizes the document and the converts alll tokens to lowercase
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
# The stemmed tokens are then joined together into a string with spaces between them and returned as the output of the function
    return " ".join(tokens)

# Takes in two documents 
def compare_files(doc1, doc2, ngram, stem=True):

# And applies stemming to them if stem is set to True
    doc1_stem = stemming(doc1, stem=stem)
    doc2_stem = stemming(doc2, stem=stem)
 # Generate document vectors using TfidfVectorizer with ngram   
    vectorizer = TfidfVectorizer(ngram_range=(
        ngram, ngram), tokenizer=word_tokenize) 
    vectorize_doc1 = vectorizer.fit_transform([doc1_stem])
    vectorize_doc2 = vectorizer.transform([doc2_stem])
#  Returns the similarity score
    return cosine_similarity(vectorize_doc1, vectorize_doc2)[0][0]



def read_documents():
    # Performs pairwise document comparision between all suspicious and original documents and 
    # generates a similarity matrix containing the similarity scrores between each pair of documents
    similarities = np.zeros((len(suspicious_files), len(original_files)))
    # The first loop iterates each suspicious document, and the second loop
    # iterates over each original document
    
    #For each pair of documents, the text content of each document is read in and passsed
    # to the function to calculate the similarity score
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


# Trains an SVM model for classifying suspicious files as plagiarized or not based on 
# their similarity scores with respect to the original files
def train_svm_model(similarities):
    # Creating the target vector
    # Indicates whether each suspicious file is plagiarized or not based on a 
    # threshold of 0.13 for the similarity score
    target = np.zeros((len(suspicious_files), len(original_files)))
    for i in range(len(suspicious_files)):
        target[i, :] = (similarities[i, :] > 0.13).astype(int)
    
    # Creates the feature matrix containing  the similarity sccores
    #flattened  into a single column.
    features = similarities.reshape(-1, 1)
    
    # The function trains the SVM model with a linear kernel using the feature matrix
    #  and the target vector.
    clf = svm.SVC(kernel='linear')
    clf.fit(features, target.reshape(-1))

    # The trained model is returned as the output of the function
    return clf

# Takes in the trained SVM model and a matrix of similarities 
def predict_plagiarism(clf, similarities):
    # Creates a feature matrix by reshaping the similarities matrix into a 1D array
    features = similarities.reshape(-1, 1)
    
    #  It uses the SVM model to predict the target vector using the feature matrix
    target = clf.predict(features)
    # The predicted target vector is reshaped to have the same dimensions as the original
    #  similarities matrix and returned
    target = target.reshape(len(suspicious_files), len(original_files))
    
    return target
