from AUC.check_plagarism import suspicious_files, original_files


def area_under_curve(true_negative, true_positive, false_negative, false_positive):
    tpr = (true_positive/(true_positive+false_negative))
    fpr = (false_positive/(false_positive+true_negative))
    auc = ((1+tpr-fpr)/2)
    return auc


def AUC_calc():
    tp, fp, tn, fn = 0, 0, 0, 0
    confirmed_plagiarized_files = [
        "FID-01.txt", "FID-02.txt", "FID-05.txt", "FID-06.txt", "FID-15.txt"]
    for i in range(len(suspicious_files)):
        if suspicious_files[i] in plagiarized_files:
            for j in range(len(original_files)):
                if similarities[i, j] >= threshold:
                    if suspicious_files[i].split('.')[0] == original_files[j].split('.')[0]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if suspicious_files[i].split('.')[0] == original_files[j].split('.')[0]:
                        fn += 1
                    else:
                        tn += 1

    # Calculate AUC
    auc = area_under_curve(tn, tp, fn, fp)
    return auc


print(AUC_calc())
