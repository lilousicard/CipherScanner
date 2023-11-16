def create_cipher_alphabet(key):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Remove duplicate letters from the key and convert it to uppercase
    unique_key = ''.join(sorted(set(key.upper()), key=key.upper().index))

    # Create the cipher alphabet starting with the unique key
    cipher_alphabet = unique_key + ''.join([ch for ch in alphabet if ch not in unique_key])
    return cipher_alphabet


def encrypt(plaintext, cipher_alphabet):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    translation_table = str.maketrans(alphabet, cipher_alphabet)

    # Encrypt the plaintext
    ciphertext = plaintext.upper().translate(translation_table)
    return ciphertext


def decrypt(ciphertext, cipher_alphabet):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    translation_table = str.maketrans(cipher_alphabet, alphabet)

    # Decrypt the ciphertext
    plaintext = ciphertext.upper().translate(translation_table)
    return plaintext

def append_to_file(cipher):
    with open('data/data.txt', 'a') as file:
        file.write('<'+ cipher + '>' + '<' + str(2) + '>' + '\n')

# Accepting the key from the user
key = input("Enter the key: ")
cipher_alphabet = create_cipher_alphabet(key)

# Accepting the plaintext from the user
plaintext = input("Enter the plaintext: ")

ciphertext = encrypt(plaintext, cipher_alphabet)
print(f"Ciphertext: {ciphertext}")
append_to_file(ciphertext)

# Decrypting back
decrypted_text = decrypt(ciphertext, cipher_alphabet)
print(f"Decrypted Text: {decrypted_text}")
