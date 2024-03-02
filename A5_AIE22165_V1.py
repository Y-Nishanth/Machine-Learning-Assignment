import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
#A1
# Read CSV file into a DataFrame
data = pd.read_csv("archive/Data/features_30_sec.csv")

# Drop the first column (assuming it contains filenames)
data = data.drop(columns=["filename"])

# Assuming the last column is the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Predictions on training set
y_train_pred = classifier.predict(X_train)

# Predictions on test set
y_test_pred = classifier.predict(X_test)

# Confusion matrix
confusion_mat_train = confusion_matrix(y_train, y_train_pred)
confusion_mat_test = confusion_matrix(y_test, y_test_pred)

# Performance metrics
precision_train = precision_score(y_train, y_train_pred, average='micro')
recall_train = recall_score(y_train, y_train_pred, average='micro')
f1_train = f1_score(y_train, y_train_pred, average='micro')

precision_test = precision_score(y_test, y_test_pred, average='micro')
recall_test = recall_score(y_test, y_test_pred, average='micro')
f1_test = f1_score(y_test, y_test_pred, average='micro')

print("Confusion Matrix (Training):")
print(confusion_mat_train)
print("Precision (Training):", precision_train)
print("Recall (Training):", recall_train)
print("F1-Score (Training):", f1_train)

print("\nConfusion Matrix (Test):")
print(confusion_mat_test)
print("Precision (Test):", precision_test)
print("Recall (Test):", recall_test)
print("F1-Score (Test):", f1_test)

#A2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
df = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='Purchase data')

# Create a binary classification target variable
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Encode the categorical target variable
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

# Extracting features and target variable
X = df.iloc[:, 1:4]  # Features: Candies, Mangoes, Milk Packets
y = df['Category']  # Numerical Target: 1 for RICH, 0 for POOR

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Creating and training the K-NN classifier model
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_std, y_train)

# Predictions on training set
y_train_pred = knn_classifier.predict(X_train_std)

# Predictions on test set
y_test_pred = knn_classifier.predict(X_test_std)

# Classification metrics
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Classification Metrics (Training):")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAE:", mae_train)
print("R2 Score:", r2_train)

print("\nClassification Metrics (Test):")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("R2 Score:", r2_test)

#A3
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
X_train_a3 = np.random.uniform(1, 10, (20, 2))
y_train_a3 = np.random.choice([0, 1], 20)

# Separate data points for class 0 and class 1
class_0_points = X_train_a3[y_train_a3 == 0]
class_1_points = X_train_a3[y_train_a3 == 1]

# Scatter plot
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], color='blue', label='Class 0')
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], color='red', label='Class 1')

plt.title("Scatter Plot of Training Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#A4
from sklearn.neighbors import KNeighborsClassifier

# Generate test set data
x_test_values = np.arange(0, 10.1, 0.1)
y_test_values = np.arange(0, 10.1, 0.1)

X_test_a4 = np.array(np.meshgrid(x_test_values, y_test_values)).T.reshape(-1, 2)

# Classify using kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_a3, y_train_a3)
y_test_pred_a4 = knn_classifier.predict(X_test_a4)

# Scatter plot of test data output
plt.scatter(X_test_a4[:, 0], X_test_a4[:, 1], c=y_test_pred_a4, cmap=plt.cm.Paired)

plt.title("Scatter Plot of Test Data Output (kNN, k=3)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#A5
k_values = [1, 5, 10]

for k in k_values:
    knn_classifier_a5 = KNeighborsClassifier(n_neighbors=k)
    knn_classifier_a5.fit(X_train_a3, y_train_a3)
    y_test_pred_a5 = knn_classifier_a5.predict(X_test_a4)

    plt.scatter(X_test_a4[:, 0], X_test_a4[:, 1], c=y_test_pred_a5, cmap=plt.cm.Paired)
    plt.title(f"Scatter Plot of Test Data Output (kNN, k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
