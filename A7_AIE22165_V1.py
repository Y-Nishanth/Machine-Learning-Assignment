import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier

# Read data from CSV file
df = pd.read_csv('features_3_sec.csv')

# Encode categorical variables


# Encode the categorical column 'label'

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
target_column = 'label'
# Split the data into features (X) and target variable (y)
X = df.drop(['filename', target_column], axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)




# Create Random Forest classifier and fit it to the training data
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_rf = random_forest_classifier.predict(X_train)
y_test_pred_rf = random_forest_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_rf = accuracy_score(y_train, y_train_pred_rf)
accuracy_y_rf = accuracy_score(y_test, y_test_pred_rf)

# precision, recall, and F1 score for training data
precision_X_rf = precision_score(y_train, y_train_pred_rf,average='weighted')
recall_X_rf = recall_score(y_train, y_train_pred_rf,average='weighted')
f1_X_rf = f1_score(y_train, y_train_pred_rf,average='weighted')

# precision, recall, and F1 score for testing data
precision_y_rf = precision_score(y_test, y_test_pred_rf,average='weighted')
recall_y_rf = recall_score(y_test, y_test_pred_rf,average='weighted')
f1_y_rf = f1_score(y_test, y_test_pred_rf,average='weighted')

print("Accuracy (Training Data):", accuracy_X_rf)
print("Precision (Training Data):", precision_X_rf)
print("Recall (Training Data):", recall_X_rf)
print("F1 Score (Training Data):", f1_X_rf)

print("\nAccuracy (Testing Data):", accuracy_y_rf)
print("Precision (Testing Data):", precision_y_rf)
print("Recall (Testing Data):", recall_y_rf)
print("F1 Score (Testing Data):", f1_y_rf)





# Create Gradient Boosting classifier and fit it to the training data
gradient_boosting_classifier = GradientBoostingClassifier(random_state=42)
gradient_boosting_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_gb = gradient_boosting_classifier.predict(X_train)
y_test_pred_gb = gradient_boosting_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_gb = accuracy_score(y_train, y_train_pred_gb)
accuracy_y_gb = accuracy_score(y_test, y_test_pred_gb)

#precision, recall, and F1 score for training data
precision_X_gb = precision_score(y_train, y_train_pred_gb,average='weighted')
recall_X_gb = recall_score(y_train, y_train_pred_gb,average='weighted')
f1_X_gb = f1_score(y_train, y_train_pred_gb,average='weighted')

#precision, recall, and F1 score for testing data
precision_y_gb = precision_score(y_test, y_test_pred_gb,average='weighted')
recall_y_gb = recall_score(y_test, y_test_pred_gb,average='weighted')
f1_y_gb = f1_score(y_test, y_test_pred_gb,average='weighted')

print("Accuracy (Training Data):", accuracy_X_gb)
print("Precision (Training Data):", precision_X_gb)
print("Recall (Training Data):", recall_X_gb)
print("F1 Score (Training Data):", f1_X_gb)

print("\nAccuracy (Testing Data):", accuracy_y_gb)
print("Precision (Testing Data):", precision_y_gb)
print("Recall (Testing Data):", recall_y_gb)
print("F1 Score (Testing Data):", f1_y_gb)




# Create Decision Tree classifier and fit it to the training data
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_dt = decision_tree_classifier.predict(X_train)
y_test_pred_dt = decision_tree_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_dt = accuracy_score(y_train, y_train_pred_dt)
accuracy_y_dt = accuracy_score(y_test, y_test_pred_dt)

# precision, recall, and F1 score for training data
precision_X_dt = precision_score(y_train, y_train_pred_dt,average='weighted')
recall_X_dt = recall_score(y_train, y_train_pred_dt,average='weighted')
f1_X_dt = f1_score(y_train, y_train_pred_dt,average='weighted')

#precision, recall, and F1 score for testing data
precision_y_dt = precision_score(y_test, y_test_pred_dt,average='weighted')
recall_y_dt = recall_score(y_test, y_test_pred_dt,average='weighted')
f1_y_dt = f1_score(y_test, y_test_pred_dt,average='weighted')

print("Accuracy (Training Data):", accuracy_X_dt)
print("Precision (Training Data):", precision_X_dt)
print("Recall (Training Data):", recall_X_dt)
print("F1 Score (Training Data):", f1_X_dt)

print("\nAccuracy (Testing Data):", accuracy_y_dt)
print("Precision (Testing Data):", precision_y_dt)
print("Recall (Testing Data):", recall_y_dt)
print("F1 Score (Testing Data):", f1_y_dt)




# Create SVM classifier and fit it to the training data
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_svm = svm_classifier.predict(X_train)
y_test_pred_svm = svm_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_svm = accuracy_score(y_train, y_train_pred_svm)
accuracy_y_svm = accuracy_score(y_test, y_test_pred_svm)

#precision, recall, and F1 score for training data
precision_X_svm = precision_score(y_train, y_train_pred_svm,average='weighted')
recall_X_svm = recall_score(y_train, y_train_pred_svm,average='weighted')
f1_X_svm = f1_score(y_train, y_train_pred_svm,average='weighted')

#precision, recall, and F1 score for testing data
precision_y_svm = precision_score(y_test, y_test_pred_svm,average='weighted')
recall_y_svm = recall_score(y_test, y_test_pred_svm,average='weighted')
f1_y_svm = f1_score(y_test, y_test_pred_svm,average='weighted')

print("Accuracy (Training Data):", accuracy_X_svm)
print("Precision (Training Data):", precision_X_svm)
print("Recall (Training Data):", recall_X_svm)
print("F1 Score (Training Data):", f1_X_svm)

print("\nAccuracy (Testing Data):", accuracy_y_svm)
print("Precision (Testing Data):", precision_y_svm)
print("Recall (Testing Data):", recall_y_svm)
print("F1 Score (Testing Data):", f1_y_svm)




# Create Naive Bayes classifier and fit it to the training data
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_nb = naive_bayes_classifier.predict(X_train)
y_test_pred_nb = naive_bayes_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_nb = accuracy_score(y_train, y_train_pred_nb)
accuracy_y_nb = accuracy_score(y_test, y_test_pred_nb)

#precision, recall, and F1 score for training data
precision_X_nb = precision_score(y_train, y_train_pred_nb,average='weighted')
recall_X_nb = recall_score(y_train, y_train_pred_nb,average='weighted')
f1_X_nb = f1_score(y_train, y_train_pred_nb,average='weighted')

# precision, recall, and F1 score for testing data
precision_y_nb = precision_score(y_test, y_test_pred_nb,average='weighted')
recall_y_nb = recall_score(y_test, y_test_pred_nb,average='weighted')
f1_y_nb = f1_score(y_test, y_test_pred_nb,average='weighted')

print("Accuracy (Training Data):", accuracy_X_nb)
print("Precision (Training Data):", precision_X_nb)
print("Recall (Training Data):", recall_X_nb)
print("F1 Score (Training Data):", f1_X_nb)

print("\nAccuracy (Testing Data):", accuracy_y_nb)
print("Precision (Testing Data):", precision_y_nb)
print("Recall (Testing Data):", recall_y_nb)
print("F1 Score (Testing Data):", f1_y_nb)




# Create CatBoost classifier and fit it to the training data
catboost_classifier = CatBoostClassifier(random_state=42)
catboost_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_cb = catboost_classifier.predict(X_train)
y_test_pred_cb = catboost_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_cb = accuracy_score(y_train, y_train_pred_cb)
accuracy_y_cb = accuracy_score(y_test, y_test_pred_cb)

# Compute precision, recall, and F1 score for training data
precision_X_cb = precision_score(y_train, y_train_pred_cb,average='weighted')
recall_X_cb = recall_score(y_train, y_train_pred_cb,average='weighted')
f1_X_cb = f1_score(y_train, y_train_pred_cb,average='weighted')

# Compute precision, recall, and F1 score for testing data
precision_y_cb = precision_score(y_test, y_test_pred_cb,average='weighted')
recall_y_cb = recall_score(y_test, y_test_pred_cb,average='weighted')
f1_y_cb = f1_score(y_test, y_test_pred_cb,average='weighted')

print("Accuracy (Training Data):", accuracy_X_cb)
print("Precision (Training Data):", precision_X_cb)
print("Recall (Training Data):", recall_X_cb)
print("F1 Score (Training Data):", f1_X_cb)

print("\nAccuracy (Testing Data):", accuracy_y_cb)
print("Precision (Testing Data):", precision_y_cb)
print("Recall (Testing Data):", recall_y_cb)
print("F1 Score (Testing Data):", f1_y_cb)




# Create AdaBoost classifier and fit it to the training data
adaboost_classifier = AdaBoostClassifier(random_state=42)
adaboost_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_ab = adaboost_classifier.predict(X_train)
y_test_pred_ab = adaboost_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_ab = accuracy_score(y_train, y_train_pred_ab)
accuracy_y_ab = accuracy_score(y_test, y_test_pred_ab)

# Compute precision, recall, and F1 score for training data
precision_X_ab = precision_score(y_train, y_train_pred_ab,average='weighted')
recall_X_ab = recall_score(y_train, y_train_pred_ab,average='weighted')
f1_X_ab = f1_score(y_train, y_train_pred_ab,average='weighted')

# Compute precision, recall, and F1 score for testing data
precision_y_ab = precision_score(y_test, y_test_pred_ab,average='weighted')
recall_y_ab = recall_score(y_test, y_test_pred_ab,average='weighted')
f1_y_ab = f1_score(y_test, y_test_pred_ab,average='weighted')

print("Accuracy (Training Data):", accuracy_X_ab)
print("Precision (Training Data):", precision_X_ab)
print("Recall (Training Data):", recall_X_ab)
print("F1 Score (Training Data):", f1_X_ab)

print("\nAccuracy (Testing Data):", accuracy_y_ab)
print("Precision (Testing Data):", precision_y_ab)
print("Recall (Testing Data):", recall_y_ab)
print("F1 Score (Testing Data):", f1_y_ab)



# Create XGBoost classifier and fit it to the training data
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred_xgb = xgb_classifier.predict(X_train)
y_test_pred_xgb = xgb_classifier.predict(X_test)

# Accuracy for training and testing data
accuracy_X_xgb = accuracy_score(y_train, y_train_pred_xgb)
accuracy_y_xgb = accuracy_score(y_test, y_test_pred_xgb)


# Compute precision, recall, and F1 score for training data
precision_X_xgb = precision_score(y_train, y_train_pred_xgb,average='weighted')
recall_X_xgb = recall_score(y_train, y_train_pred_xgb,average='weighted')
f1_X_xgb = f1_score(y_train, y_train_pred_xgb,average='weighted')

# Compute precision, recall, and F1 score for testing data
precision_y_xgb = precision_score(y_test, y_test_pred_xgb,average='weighted')
recall_y_xgb = recall_score(y_test, y_test_pred_xgb,average='weighted')
f1_y_xgb = f1_score(y_test, y_test_pred_xgb,average='weighted')

print("Accuracy (Training Data):", accuracy_X_xgb)
print("Precision (Training Data):", precision_X_xgb)
print("Recall (Training Data):", recall_X_xgb)
print("F1 Score (Training Data):", f1_X_xgb)

print("\nAccuracy (Testing Data):", accuracy_y_xgb)
print("Precision (Testing Data):", precision_y_xgb)
print("Recall (Testing Data):", recall_y_xgb)
print("F1 Score (Testing Data):", f1_y_xgb)





# Define the parameter distribution for the RandomizedSearchCV
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
    'activation': ['logistic', 'relu', 'tanh'],
    'solver': ['adam', 'sgd'],  # Try different solvers
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 5000, 10000, 15000]
}


# Create a RandomizedSearchCV object for hyperparameter tuning
random_search = RandomizedSearchCV(MLPClassifier(random_state=1, max_iter=2000), param_distributions=param_dist, n_iter=20, cv=5, random_state=42)


# Perform the randomized search over the parameter space
random_search.fit(X_train, y_train)

# Get the best parameters and best score found during the search
best_params = random_search.best_params_
best_score = random_search.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)


random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Predictions on testing data
y_test_pred_rf = random_forest_classifier.predict(X_test)

# Accuracy for testing data
accuracy_test = accuracy_score(y_test, y_test_pred_rf)
print("\nAccuracy (Testing Data):", accuracy_test)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(random_forest_classifier)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# Visualize feature importances
shap.summary_plot(shap_values, X_test)