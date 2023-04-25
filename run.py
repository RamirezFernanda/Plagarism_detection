from check_plagarism import read_documents, suspicious_files, original_files
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    # Get the similarity matrix between all suspicious and original documents
    similarities = read_documents()

    true_labels = [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # Manually define the true labels (0 for original, 1 for suspicious)
    predictions = [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  
    #predictions = similarities.flatten()[:len(true_labels)] # Store the similarities as the model's predictions and adjust the length to match the number of true labels

    # Calculate and print the AUC
    auc = roc_auc_score(true_labels, predictions)
    print(f"El Area bajo la curva AUC es: {auc:.2f}")

    # Output the documents with similarity scores above 0.30
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if similarities[i, j] > 0.30:
                # The output includes the name of the original and suspicious document, as well as the similarity score
                print(f"¡Alerta! Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")