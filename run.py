'''
Authors: Maria Fernanda Ramirez Barragan,
         Melissa Garduño Ruiz
Creation date: 23/04/23  dd/mm/yy
'''

from Algorithms.check_plagarism import read_documents, suspicious_files, original_files
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    # Get the similarity matrix between all suspicious and original documents
    similarities = read_documents()
    # Output the documents with similarity scores above 0.30
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if similarities[i, j] > 0.30:
                # The output includes the name of the original and suspicious document, as well as the similarity score
                print(f"¡Alerta! Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")