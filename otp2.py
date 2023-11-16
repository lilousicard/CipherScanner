import random
import string

#This python code generate one time pad cipher that are only in letters
def generate_key(message):
    """ Generate a random key with the same length as the message. """
    return ''.join(random.choices(string.ascii_letters, k=len(message)))

def encrypt_message(message, key):
    """ Encrypt the message using a custom One-Time Pad cipher. """
    ciphertext = ''
    for m, k in zip(message, key):
        if m.isalpha():
            shift = ord(k.lower()) - ord('a')
            if m.isupper():
                ciphertext += chr((ord(m) - ord('A') + shift) % 26 + ord('A'))
            else:
                ciphertext += chr((ord(m) - ord('a') + shift) % 26 + ord('a'))
        else:
            ciphertext += m
    return ciphertext

def decrypt_message(ciphertext, key):
    """ Decrypt the message using a custom One-Time Pad cipher. """
    plaintext = ''
    for c, k in zip(ciphertext, key):
        if c.isalpha():
            shift = ord(k.lower()) - ord('a')
            if c.isupper():
                plaintext += chr((ord(c) - ord('A') - shift) % 26 + ord('A'))
            else:
                plaintext += chr((ord(c) - ord('a') - shift) % 26 + ord('a'))
        else:
            plaintext += c
    return plaintext

def main():
    # Example usage
    message = input("Enter the message: ")
    key = generate_key(message)
    print("Key:", key)
    # Encrypt the message
    ciphertext = encrypt_message(message, key)
    print("Encrypted:", ciphertext)
    append_to_file(ciphertext)
    # Decrypt the ciphertext
    decrypted_text = decrypt_message(ciphertext, key)
    print("Decrypted:", decrypted_text)

def append_to_file(cipher):
    with open('data/data.txt', 'a') as file:
        file.write('<'+ cipher + '>' + '<' + str(3) + '>' + '\n')

if __name__ == "__main__":
    main()