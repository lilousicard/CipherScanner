import os
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
    with open(filename, 'r') as file:
        content = file.read()
    filtered_content = ''.join(char.upper() for char in content if char.isalpha())
    return filtered_content


def count_letters(string):
    letter_counts = {}
    for char in string:
        if char.isalpha():
            if char in letter_counts:
                letter_counts[char] += 1
            else:
                letter_counts[char] = 1
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

    # Calculate the probabilities

    return sequence_counts


def display_heatmap(sequence_probabilities):
    # Convert the 2D array to a pandas DataFrame
    df = pd.DataFrame(sequence_probabilities)

    # Set the index and columns of the DataFrame to the letters of the alphabet
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    df.index = letters
    df.columns = letters

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='YlGnBu')
    plt.show()


def display_sequence_probabilities(sequence_probabilities):
    # Convert the 2D array to a pandas DataFrame
    df = pd.DataFrame(sequence_probabilities)

    # Set the index and columns of the DataFrame to the letters of the alphabet
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    df.index = letters
    df.columns = letters

    # Print the DataFrame to the console
    print(df)


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


def main():
    filename = get_filename()
    filtered_content = read_and_filter(filename)
    letter_counts = count_letters(filtered_content)
    plot_letter_counts(letter_counts)
    sequence_probabilities = calculate_sequence_probabilities(filtered_content)
    display_sequence_probabilities(sequence_probabilities)
    display_heatmap(sequence_probabilities)


main()
