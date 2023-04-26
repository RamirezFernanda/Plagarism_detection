from sklearn.metrics import roc_auc_score
from check_plagarism import read_documents, suspicious_files, original_files

# Manually define the labels names
labels_name = ['FID-01', 'FID-02', 'FID-03', 'FID-04', 'FID-05', 'FID-06', 'FID-07', 'FID-08', 'FID-09', 'FID-10', 'FID-11', 'FID-12', 'FID-13', 'FID-14','FID-15'] 
# Manually define the true labels (0 for original, 1 for suspicious)
true_labels = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  

# Function that checks if similarity scores is > than 0.30, if so, append the plagarisim file name
def check_files():
    # Get the similarity matrix between all suspicious and original documents
    similarities = read_documents()
    plag_docs_pred = []
    # Output the documents with similarity scores above 0.30
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            if similarities[i, j] > 0.30:
                # If so, append plagiarized_file name to plag_docs_pred
                if plagiarized_file.split('.')[0] not in plag_docs_pred:
                    # If plagiarized_file name is already in plag_docs_pred, it doesn't append it twice
                    plag_docs_pred.append(plagiarized_file.split('.')[0])
    return plag_docs_pred

#Function that gets the predictions given plag_docs_pred of "check_files()"
def get_predictions():
    predictions = []
    plag_docs_pred = check_files()
    # Check if the plag_docs_pred names are in labels_name
    for name in labels_name:
        # If so, append 1 to predictions empty list
        if name in plag_docs_pred:
            predictions.append(1)
        # If not, append 0 to predictions empty list
        else:
            predictions.append(0)
    # Returns the list with the predictions
    return predictions

predictions = get_predictions()
# Give AUC score
auc = roc_auc_score(true_labels, predictions)
print('------------------------------------------')
print(f"El Area bajo la curva AUC es: {auc:.2f}")
print('------------------------------------------')