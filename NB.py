from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from extract_data import count_letters, calculate_sequence_probabilities, read_and_filter
import matplotlib.pyplot as plt

# Assuming ciphers is a dictionary where the keys are the ciphers and the values are the encryption method
filename = "data/data.txt"
ciphers = read_and_filter(filename)

# Prepare the data for the Naive Bayes
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

# Create a label vector (The cipher type is the label)
y = list(ciphers.values())

# Split the data into training and testing sets
X_train_both, X_test_both, y_train, y_test = train_test_split(X_both, y, test_size=0.25, random_state=42)
X_train_letters, X_test_letters = train_test_split(X_letters, test_size=0.25, random_state=42)[:2]
X_train_bigrams, X_test_bigrams = train_test_split(X_bigrams, test_size=0.25, random_state=42)[:2]

# Train the Naive Bayes
clf_both = GaussianNB()
clf_both.fit(X_train_both, y_train)

clf_letters = GaussianNB()
clf_letters.fit(X_train_letters, y_train)

clf_bigrams = GaussianNB()
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

# Plotting the results and the confusion matrix
accuracy = [metrics.accuracy_score(y_test, y_pred_both), metrics.accuracy_score(y_test, y_pred_letters),
            metrics.accuracy_score(y_test, y_pred_bigrams)]
recall = [metrics.recall_score(y_test, y_pred_both, average='weighted'),
          metrics.recall_score(y_test, y_pred_letters, average='weighted'),
          metrics.recall_score(y_test, y_pred_bigrams, average='weighted')]

# X-axis labels
labels = ['Both Features', 'Letter Counts', 'Bigram Probabilities']

# Create figure and axes
fig, ax = plt.subplots(2, 1, figsize=(15, 15))

# Create accuracy subplot
ax[0].bar(labels, accuracy)
ax[0].set_title('Accuracy for Different Feature Sets using NB')
ax[0].set_xlabel('Feature Set')
ax[0].set_ylabel('Accuracy')

# Create recall subplot
ax[1].bar(labels, recall)
ax[1].set_title('Recall for Different Feature Sets using NB')
ax[1].set_xlabel('Feature Set')
ax[1].set_ylabel('Recall')

# Show the plot
plt.tight_layout()
plt.show()

# Calculate confusion matrices for each model
cm_both = confusion_matrix(y_test, y_pred_both)
cm_letters = confusion_matrix(y_test, y_pred_letters)
cm_bigrams = confusion_matrix(y_test, y_pred_bigrams)

# Create figure and axes
fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# Create confusion matrix heatmaps
sns.heatmap(cm_both, annot=True, ax=ax[0], cmap='Blues', xticklabels=['DT', 'SS', 'OTP'],
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