from check_plagarism import suspicious_files, original_files, read_documents


def area_under_curve(true_negative, true_positive, false_negative, false_positive):
    try: 
        tpr = (true_positive/(true_positive+false_negative))
        fpr = (false_positive/(false_positive+true_negative))
        auc = ((1+tpr-fpr)/2)
        return auc
    except ZeroDivisionError:
        return 0


def AUC_calc():
    similarities = read_documents()
    tp, fp, tn, fn = 0, 0, 0, 0
    plagiarized_files = [
        "FID-01.txt", "FID-02.txt", "FID-05.txt", "FID-06.txt", "FID-15.txt"]
    for i in range(len(suspicious_files)):
        if suspicious_files[i] in plagiarized_files:
            for j in range(len(original_files)):
                if similarities[i, j] >= 0.30:
                    if suspicious_files[i].split('.')[0] in suspicious_files:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if suspicious_files[i].split('.')[0] in suspicious_files:
                        fn += 1
                    else:
                        tn += 1
    print(tp, tn, fp, fn)

    # Calculate AUC
    auc = area_under_curve(tn, tp, fn, fp)
    return auc


print(AUC_calc())
