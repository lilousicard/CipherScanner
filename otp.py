import random
import string

import onetimepad


def encrypt_message(message, key):
    """ Encrypt the message using One-Time Pad cipher. """
    return onetimepad.encrypt(message, key)


def decrypt_message(ciphertext, key):
    """ Decrypt the message using One-Time Pad cipher. """
    return onetimepad.decrypt(ciphertext, key)


def main():
    # Example usage
    message = "MeetingPostponedUntilThreepm"
    key = ''.join(random.choices(string.ascii_letters, k=len(message)))  # The key should be random and at least as long as the message
    # Encrypt the message
    ciphertext = encrypt_message(message, key)
    print("Encrypted:", ciphertext)

    # Decrypt the ciphertext
    decrypted_text = decrypt_message(ciphertext, key)
    print("Decrypted:", decrypted_text)
    print("Key:", key)


if __name__ == "__main__":
    main()
