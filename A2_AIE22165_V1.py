import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# To calculate Euclidean distance
def euclid_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

# To calculate Manhattan distance
def manhattan_dist(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

# To implement k-NN classifier using Euclidean distance
def knn(training_data, test, k):
    distances = [euclid_dist(test, x) for x in training_data]
    sorted_indices = np.argsort(distances)
    k_nearest_neighbors = sorted_indices[:k]
    return k_nearest_neighbors

# To implement k-NN classifier using Manhattan distance
def knn_man(training_data, test, k):
    distances = [manhattan_dist(test, x) for x in training_data]
    sorted_indices = np.argsort(distances)
    k_nearest_neighbors = sorted_indices[:k]
    return k_nearest_neighbors

# To convert categorical variables to numeric using label encoding
def label_encoding(data):
    unique_labels = list(set(data))
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    encoded_data = [label_dict[label] for label in data]
    return encoded_data

# To convert categorical variables to numeric using One-Hot encoding
def onehot_coding(variables):
    unique_categories = list(set(variables))
    category_to_index = {category: i for i, category in enumerate(unique_categories)}
    encoded_labels = []
    for var in variables:
        binary_vector = [0] * len(unique_categories)
        binary_vector[category_to_index[var]] = 1
        encoded_labels.append(binary_vector)
    return encoded_labels, category_to_index

# Load ARFF file
data, meta = arff.loadarff("3sources_bbc1000.arff")

# Convert ARFF data to a numpy array
dataset = np.array(data.tolist(), dtype=float)

# Assume the last column is the target variable
X = dataset[:, :-1]
y = dataset[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to predict labels for the test set using Euclidean distance
def predict_labels(X_train, y_train, test, k):
    k_nearest_neighbors = knn(X_train, test, k)
    k_nearest_labels = y_train[k_nearest_neighbors]
    predicted_label = np.argmax(np.bincount(k_nearest_labels.astype(int)))
    return predicted_label

# Predict labels for the test set using Euclidean distance
y_pred_euclidean = [predict_labels(X_train, y_train, test, k=15) for test in X_test]

# Function to predict labels for the test set using Manhattan distance
def predict_labels_manhattan(X_train, y_train, test, k):
    k_nearest_neighbors = knn_man(X_train, test, k)
    k_nearest_labels = y_train[k_nearest_neighbors]
    predicted_label = np.argmax(np.bincount(k_nearest_labels.astype(int)))
    return predicted_label

# Predict labels for the test set using Manhattan distance
y_pred_manhattan = [predict_labels_manhattan(X_train, y_train, test, k=15) for test in X_test]

# Evaluate the performance for Euclidean distance
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
conf_matrix_euclidean = confusion_matrix(y_test, y_pred_euclidean)
classification_rep_euclidean = classification_report(y_test, y_pred_euclidean)

# Evaluate the performance for Manhattan distance
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
conf_matrix_manhattan = confusion_matrix(y_test, y_pred_manhattan)
classification_rep_manhattan = classification_report(y_test, y_pred_manhattan)

# Print evaluation metrics for Euclidean distance
print("Euclidean Distance:")
print("Accuracy:", accuracy_euclidean)
print("Confusion Matrix:\n", conf_matrix_euclidean)
print("Classification Report:\n", classification_rep_euclidean)

# Print evaluation metrics for Manhattan distance
print("\nManhattan Distance:")
print("Accuracy:", accuracy_manhattan)
print("Confusion Matrix:\n", conf_matrix_manhattan)
print("Classification Report:\n", classification_rep_manhattan)

# Predict labels for the test set using Euclidean distance and print predicted class
for i, test in enumerate(X_test):
    predicted_label = predict_labels(X_train, y_train, test, k=15)
    print(f"Test {i + 1}: Predicted class - {int(predicted_label)}")

# Predict labels for the test set using Manhattan distance and print predicted class
for i, test in enumerate(X_test):
    predicted_label = predict_labels_manhattan(X_train, y_train, test, k=15)
    print(f"Test {i + 1}: Predicted class - {int(predicted_label)}")

# One-Hot Encoding
one_hot_encoded_data = onehot_coding(y)
print("One-Hot Encoded Data:")
print(one_hot_encoded_data)

# Label Encoding
label_encoded_data = label_encoding(y)
print("Label Encoded Data:")
print(label_encoded_data)