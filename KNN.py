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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)  # You can experiment with n_neighbors
knn.fit(X_train, y_train) # Fit on the training set

# Predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')  # Plotting confusion matrix
plt.xlabel('Predicted')
plt.ylabel('True')
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