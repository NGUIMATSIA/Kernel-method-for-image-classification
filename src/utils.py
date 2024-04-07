import matplotlib.pyplot as plt
import numpy as np


def inspect_label(labels):
  unique_elements, counts = np.unique(labels, return_counts=True)
  plt.bar(unique_elements, counts, color='skyblue')
  plt.show()

def scaler(train_set, val_set = None, test_set = None):
    mu_X = train_set.mean()
    sigma_X =train_set.std()
    return (train_set - mu_X)/sigma_X, (val_set - mu_X)/sigma_X, (test_set - mu_X)/sigma_X

def create_Submissioncsv(y_pret):
    f = open("Yte.csv", "w")
    f.write("Id,Prediction\n")
    for n in range(len(y_pret)):
        f.write("{},{}\n".format(int(n+1),y_pret[n]))
    f.close()

def train_test_split(data, labels, test_size=0.3, shuffle=True):
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
    test_samples = int(len(data) * test_size)
    test_data = data[:test_samples]
    test_labels = labels[:test_samples]
    train_data = data[test_samples:]
    train_labels = labels[test_samples:]
    return train_data, test_data, train_labels, test_labels


def cross_validation(model, hog_features_train, sift_features_train, train_label, n_folds=4):
    fold_accuracies = []
    indices = np.arange(len(hog_features_train))
    np.random.shuffle(indices)
    fold_size = len(hog_features_train) // n_folds
    for i in range(n_folds):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        X_train_hog, X_test_hog = hog_features_train[train_indices], hog_features_train[test_indices]
        X_train_sift, X_test_sift = sift_features_train[train_indices], sift_features_train[test_indices]
        y_train, y_test = train_label[train_indices], train_label[test_indices]
        model.fit(X_train_hog, X_train_sift, y_train)
        predictions = model.predict(X_test_hog, X_test_sift)
        accuracy = np.mean(predictions == y_test)
        fold_accuracies.append(accuracy)
    avg_accuracy = np.mean(fold_accuracies)
    return avg_accuracy, fold_accuracies