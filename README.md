# Plagarism detection program

## Description

The project presented below seeks to detect plagiarism through the similarity of cosine and n-grams in the comparison of files with possible plagiarism and their corresponding original documents.
There is a '_check_plagarism.py_' file that contains the main code to achieve the plagarism detection, we also include
a '_run.py_' file that allows the user to execute the main code and visualize the obteined results.

## Installing

To run the '_run.py_' file, you first need to install some libraries,
to achieve that, you will need to use the next commands in
your terminal:

- pip install nltk
- pip install scikit-learn
- pip install joblib

## Running _run.py_

To run the '_run.py_' file, after installing the required libraries, you must
open a new terminal, then, execute the command:

- python3 run.py

## Add your own documents

To execute the program with your own original and suspicious files, you must add the documents
in the folder named **Detection_Files**, there, you will find **Originals**, **Plag**, and **Suspicious** folders.
Add the original documents to **Originals** folder and the suspicious ones to **Suspicious** folder, and then
execute '_run.py_' file.
