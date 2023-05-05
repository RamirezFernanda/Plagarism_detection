'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''
# Imports specific functions and variables from the module "check_plagiarism" located in the "Algorithms" directory
from Algorithms.check_plagarism import read_documents, suspicious_files, original_files, train_svm_model, predict_plagiarism


# Checks if the script is being executed as the main program
if __name__ == '__main__':

# Checks for  plagiarism by calling the read_documents function to get the similarity scores between all suspicious 
# files and all original files
    similarities = read_documents()
# Trains the SVM model using these similarity scores and makes predictions about which pairs of files are likely to be plagiarized 
    clf = train_svm_model(similarities)
# Returns a matrix of predicted plagiarism, where each element of the matrix represents the predicted label (0 or 1) for the corresponding 
# suspicious and original document pair.
    predictions = predict_plagiarism(clf, similarities)
 # Loops through all pairs of suspicious and original files and prints a message if plagiarism is detected, along with the cosine 
 # similarity score between the two documents.  
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if predictions[i, j] == 1:
                print(f"Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print('--------------------------------------------------------------------------------------------------------')