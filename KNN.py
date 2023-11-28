from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from extract_data import read_and_filter, count_letters, calculate_sequence_probabilities

filename = "data/data.txt"
ciphers = read_and_filter(filename)

X = []
X_letters = []
X_bigrams = []
for cipher in ciphers.keys():
    letter_counts = list(count_letters(cipher).values())
    bigram_probabilities = calculate_sequence_probabilities(cipher)
    bigram_probabilities = bigram_probabilities.flatten()
    X.append(letter_counts + list(bigram_probabilities))
    X_letters.append(letter_counts)
    X_bigrams.append(list(bigram_probabilities))

# Create a label vector (The cipher type is the label)
y = list(ciphers.values())

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling directly

X_bigrams_scaled = scaler.fit_transform(X_bigrams)

X_letters_scaled = scaler.fit_transform(X_letters)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

X_train_letters, X_test_letters = train_test_split(X_letters_scaled, test_size=0.25, random_state=42)[:2]
X_train_bigrams, X_test_bigrams = train_test_split(X_bigrams_scaled, test_size=0.25, random_state=42)[:2]

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)  # You can experiment with n_neighbors
knn_letters = KNeighborsClassifier(n_neighbors=3)
knn_bigrams = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) # Fit on the training set
knn_letters.fit(X_train_letters, y_train)
knn_bigrams.fit(X_train_bigrams, y_train)
# Predictions
y_pred = knn.predict(X_test)
y_pred_letters = knn_letters.predict(X_test_letters)
y_pred_bigrams = knn_bigrams.predict(X_test_bigrams)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_score_letters = accuracy_score(y_test, y_pred_letters)
accuracy_score_bigrams = accuracy_score(y_test, y_pred_bigrams)
print("Accuracy:", accuracy)
print("Accuracy for letter count:", accuracy_score_letters)
print("Accuracy for bigrams:", accuracy_score_bigrams)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_letters = confusion_matrix(y_test, y_pred_letters)
cm_bigrams = confusion_matrix(y_test, y_pred_bigrams)
# Create figure and axes
fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# Create confusion matrix heatmaps
sns.heatmap(cm, annot=True, ax=ax[0], cmap='Blues', xticklabels=['DT', 'SS', 'OTP'],
                 yticklabels=['DT', 'SS', 'OTP'])
ax[0].set_title('Confusion Matrix for Both Features')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(cm_letters, annot=True, ax=ax[1], cmap='Blues', xticklabels=['DT', 'SS', 'OTP'],
                 yticklabels=['DT', 'SS', 'OTP'])
ax[1].set_title('Confusion Matrix for Letter Counts')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

sns.heatmap(cm_bigrams, annot=True, ax=ax[2], cmap='Blues', xticklabels=['DT', 'SS', 'OTP'],
                 yticklabels=['DT', 'SS', 'OTP'])
ax[2].set_title('Confusion Matrix for Bigram Probabilities')
ax[2].set_xlabel('Predicted')
ax[2].set_ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

k_values = range(1, 26)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric='manhattan'), X_scaled, y, cv=5).mean() for k in k_values]

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Cross-Validation Accuracy for Different K Values')
plt.show()

cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric='manhattan'), X_letters_scaled, y, cv=5).mean() for k in k_values]

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Cross-Validation Accuracy for Different K Values for letter count only')
plt.show()

cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric='manhattan'), X_bigrams_scaled, y, cv=5).mean() for k in k_values]

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Cross-Validation Accuracy for Different K Values for bigram count only')
plt.show()