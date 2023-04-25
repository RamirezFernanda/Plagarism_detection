from check_plagarism import read_documents, suspicious_files, original_files
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    # Get the similarity matrix between all suspicious and original documents
    similarities = read_documents()

    true_labels = []  # to store the true labels (0 for original, 1 for suspicious)
    predictions = []  # to store the model's predictions

    # Calculate true_labels and predictions
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if similarities[i, j] > 0.30:
                true_labels.append(1)  # if similarity is greater than 0.30, the document is suspicious, so its label is 1
            else:
                true_labels.append(0)  # if similarity is less than or equal to 0.30, the document is original, so its label is 0
            predictions.append(similarities[i, j])  # save the similarity as the model's prediction

            if similarities[i, j] > 0.30:
                # The output includes the name of the original and suspicious document, as well as the similarity score
                print(
                    f"¡Alerta! Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    # Calculate and print the AUC
    auc = roc_auc_score(true_labels, predictions)
    print(f"El Area bajo la curva AUC es: {auc:.2f}")
