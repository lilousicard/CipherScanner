from sklearn import svm
from sklearn.model_selection import train_test_split
from extract_data import count_letters, calculate_sequence_probabilities, read_and_filter
from sklearn import metrics

# Assuming ciphers is a dictionary where the keys are the ciphers and the values are the letter counts
filename = "data/data.txt"
ciphers = read_and_filter(filename)

# Prepare the data for the three models
X_both = []
X_letters = []
X_bigrams = []
for cipher in ciphers.keys():
    letter_counts = list(count_letters(cipher).values())
    bigram_probabilities = calculate_sequence_probabilities(cipher)
    bigram_probabilities = bigram_probabilities.flatten()
    X_both.append(letter_counts + list(bigram_probabilities))
    X_letters.append(letter_counts)
    X_bigrams.append(list(bigram_probabilities))

# Create a label vector (this would depend on your specific use case)
y = list(ciphers.values())

# Split the data into training and testing sets
X_train_both, X_test_both, y_train, y_test = train_test_split(X_both, y, test_size=0.2, random_state=42)
# We only need X_train and X_test
X_train_letters, X_test_letters = train_test_split(X_letters, test_size=0.2, random_state=42)[:2]
X_train_bigrams, X_test_bigrams = train_test_split(X_bigrams, test_size=0.2, random_state=42)[:2]

# Train the SVMs
clf_both = svm.SVC()
clf_both.fit(X_train_both, y_train)

clf_letters = svm.SVC()
clf_letters.fit(X_train_letters, y_train)

clf_bigrams = svm.SVC()
clf_bigrams.fit(X_train_bigrams, y_train)

# Predict the labels of the test set
y_pred_both = clf_both.predict(X_test_both)
y_pred_letters = clf_letters.predict(X_test_letters)
y_pred_bigrams = clf_bigrams.predict(X_test_bigrams)

# Calculate metrics for both features
print("For both features:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_both))
print("Recall:", metrics.recall_score(y_test, y_pred_bigrams, average='weighted'))

# Calculate metrics for letter counts
print("\nFor letter counts:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_letters))
print("Recall:", metrics.recall_score(y_test, y_pred_bigrams, average='weighted'))

# Calculate metrics for bigram probabilities
print("\nFor bigram probabilities:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_bigrams))
print("Recall:", metrics.recall_score(y_test, y_pred_bigrams, average='weighted'))
