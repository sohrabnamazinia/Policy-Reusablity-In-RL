import numpy as np

def index_to_state(number):
        binary_str = bin(number)[2:]  # [2:] to remove the '0b' prefix
        # Calculate the number of zero padding needed
        padding_length = 9 - len(binary_str)
        # Pad the binary string with leading zeros
        binary_vector = '0' * padding_length + binary_str
        # Convert the binary string to a NumPy array of integers
        binary_array = np.array(list(binary_vector), dtype=int)
        return binary_array

print(index_to_state(7))