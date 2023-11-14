def create_transposition_grid(message, num_cols):
    """ Create a transposition grid based on the number of columns. """
    grid = []
    for i in range(0, len(message), num_cols):
        grid.append(message[i:i + num_cols].ljust(num_cols, '*'))  # Pad with '*' if necessary
    return grid


def transpose(grid, key):
    """ Transpose the grid based on the key. """
    num_rows = len(grid)
    transposed = [''] * num_rows
    for index, col_index in enumerate(key):
        for row in range(num_rows):
            char = grid[row][col_index]
            transposed[row] += char if char != ' ' else '_'
    return ''.join(transposed)


def encrypt(message, key1, key2):
    """ Encrypt the message using double transposition cipher. """
    message = message.replace(' ', '_')  # Replace spaces with underscores
    num_cols_key1 = len(key1)
    grid = create_transposition_grid(message, num_cols_key1)
    transposed_message = transpose(grid, key1)

    num_cols_key2 = len(key2)
    grid = create_transposition_grid(transposed_message, num_cols_key2)
    encrypted_message = transpose(grid, key2)

    return encrypted_message


def decrypt(ciphertext, key1, key2):
    """ Decrypt the message using double transposition cipher. """
    # Reverse the keys
    reversed_key1 = sorted(range(len(key1)), key=key1.__getitem__)
    reversed_key2 = sorted(range(len(key2)), key=key2.__getitem__)

    num_cols_key2 = len(key2)
    grid = create_transposition_grid(ciphertext, num_cols_key2)
    reversed_message = transpose(grid, reversed_key2)

    num_cols_key1 = len(key1)
    grid = create_transposition_grid(reversed_message, num_cols_key1)
    decrypted_message = transpose(grid, reversed_key1)

    return decrypted_message.replace('*', '').replace('_', ' ')  # Remove padding and replace underscores with spaces


def main():
    # Example usage with key1 and key2 as permutation of indices
    message = "Meeting postponed until 3pm."
    key1 = [1, 0, 2]  # Example key for first transposition
    key2 = [0, 2, 1]  # Example key for second transposition

    # Encrypt the message
    ciphertext = encrypt(message, key1, key2)
    print("Encrypted:", ciphertext)

    # Decrypt the ciphertext
    decrypted_text = decrypt(ciphertext, key1, key2)
    print("Decrypted:", decrypted_text)


if __name__ == "__main__":
    main()
