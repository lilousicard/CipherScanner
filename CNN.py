import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from extract_data import count_letters, calculate_sequence_probabilities, read_and_filter

# Assuming ciphers is a dictionary where the keys are the ciphers and the values are the encryption method
filename = "data/data.txt"
ciphers = read_and_filter(filename)

# Prepare the data for the CNN
X = []
for cipher in ciphers.keys():
    letter_counts = list(count_letters(cipher).values())
    bigram_probabilities = calculate_sequence_probabilities(cipher)
    bigram_probabilities = bigram_probabilities.flatten()
    X.append(letter_counts + list(bigram_probabilities))

# Convert list to numpy array and reshape to 3D for CNN
X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Create a label vector (The cipher type is the label)
y = list(ciphers.values())

# Convert labels to categorical
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the architecture of the CNN
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(Dense(y.shape[1], activation='softmax'))

optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Show Confusion Matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
plt.figure(figsize=(10, 10))

ax = sns.heatmap(cm, annot=True, square=True, cmap='Blues', cbar=False, xticklabels=['DT', 'SS', 'OTP'],
                 yticklabels=['DT', 'SS', 'OTP'])
ax.set_title('Confusion Matrix for Both Features')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
