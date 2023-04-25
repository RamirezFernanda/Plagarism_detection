from AUC.check_plagarism import suspicious_files, original_files, read_documents


if __name__ == '__main__':
    similarities = read_documents()
    # The results are printed out for each pair of documents
    for i, plagiarized_file in enumerate(suspicious_files):
        for j, original_file in enumerate(original_files):
            # If the similarity score between an original and a suspicious document is greater than 30%
            if similarities[i, j] >= 0.30:
                # The output includes the name of the original and suspicious document, as well as the similarity score
                print(
                    f"¡Alerta! Se ha detectado plagio en el documento {plagiarized_file}. \nLa similitud de coseno entre el documento original {original_file} y el documento plagiado {plagiarized_file} es: {similarities[i,j]*100:.2f}%")
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
