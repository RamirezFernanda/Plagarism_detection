'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''

from Algorithms.check_plagarism import read_documents, suspicious_files, original_files, train_svm_model, predict_plagiarism



if __name__ == '__main__':

    similarities = read_documents()
    clf = train_svm_model(similarities)
    predictions = predict_plagiarism(clf, similarities)
    
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if predictions[i, j] == 1:
                print(f"¡Alerta! Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print('--------------------------------------------------------------------------------------------------------')