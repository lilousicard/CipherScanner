import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_filename():
    while True:
        filename = input("Please enter the filename: ")
        if os.path.isfile(filename):
            return filename
        else:
            print("File does not exist. Please try again.")


def read_and_filter(filename):
    ciphers = {}

    # Open the file in read mode
    with open(filename, 'r') as file:
        # skip first two lines
        file.readline()
        file.readline()
        # Read each line in the file
        for line in file:
            # Strip newline and any extra whitespace
            line = line.strip()

            # Split the line to separate cipher and method
            cipher, method = line.split('><')

            # Remove the angle brackets from cipher and method
            cipher = cipher[1:]  # Remove the first '<'
            method = method[:-1]  # Remove the last '>'
            cipher = re.sub('[^A-Za-z]+', 'X', cipher)
            cipher = cipher.upper()
            # Store the cipher and method in the dictionary
            ciphers[cipher] = int(method)  # Convert method to an integer

    # Print the dictionary to verify
    # print(ciphers)
    return ciphers


def count_letters(string):
    letter_counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    total_letters = 0
    for char in string:
        if char.isalpha():
            total_letters += 1
            letter_counts[char] += 1
    for letter, count in letter_counts.items():
        letter_counts[letter] = count / total_letters
    return letter_counts


def calculate_sequence_probabilities(string):
    # Initialize a 26x26 2D array with all elements set to 0
    sequence_counts = np.zeros((26, 26))

    total_sequences = len(string) - 1

    for i in range(total_sequences):
        # Convert the letters to upper case and get their ASCII values
        first_letter = ord(string[i].upper()) - ord('A')
        second_letter = ord(string[i + 1].upper()) - ord('A')

        # Increment the corresponding element in the 2D array
        sequence_counts[first_letter][second_letter] += 1

    # Normalize the counts to get probabilities
    sequence_probabilities = normalize_sequence_probabilities(sequence_counts)

    return sequence_probabilities


def display_heatmap(sequence_probabilities):
    # Normalize the probabilities
    #total_sequences = normalize_sequence_probabilities(sequence_probabilities)
    # Convert the 2D array to a pandas DataFrame
    df = pd.DataFrame(sequence_probabilities)

    # Set the index and columns of the DataFrame to the letters of the alphabet
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    df.index = letters
    df.columns = letters

    # Create a heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(df, annot=True, cmap='YlGnBu')
    plt.show()


def display_sequence_probabilities(sequence_probabilities):
    # Convert the 2D array to a pandas DataFrame
    df = pd.DataFrame(sequence_probabilities)

    # Set the index and columns of the DataFrame to the letters of the alphabet
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    df.index = letters
    df.columns = letters

    # Create a figure and a subplot with a larger size
    fig, ax = plt.subplots(figsize=(20, 20))

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Create a table and add it to the plot
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    plt.show()


def plot_letter_counts(letter_counts):
    letters = list(letter_counts.keys())
    counts = list(letter_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(letters, counts)
    plt.xlabel('Letters')
    plt.ylabel('Counts')
    plt.title('Letter Counts')
    plt.show()


def normalize_sequence_probabilities(sequence_probabilities):
    # Sum each row and reshape the result to a column vector
    row_sums = sequence_probabilities.sum(axis=1).reshape(-1, 1)

    # Avoid division by zero by replacing zero sums with one
    row_sums[row_sums == 0] = 1

    # Divide each row of the sequence_probabilities array by the sum of that row
    normalized_probabilities = sequence_probabilities / row_sums

    return normalized_probabilities


def main():
    filename = "data/data.txt"
    ciphers = read_and_filter(filename)
    for cipher in ciphers:
        letter_count = count_letters(cipher)
        plot_letter_counts(letter_count)
        sequence_probabilities = calculate_sequence_probabilities(cipher)
        display_heatmap(sequence_probabilities)


#main()
