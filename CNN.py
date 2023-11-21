import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the architecture of the CNN
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # Change this to match the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])  # Change the loss to 'categorical_crossentropy'

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
