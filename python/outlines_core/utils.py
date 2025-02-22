import array
from typing import List

from bitarray import bitarray


def mask_to_list(mask_buffer: array.array) -> List[int]:
    """
    Converts a mask buffer into a list of token IDs where bits are set to 1.
    Caution : It can be very slow when a lot of tokens is allowed.

    Args:
        mask_buffer: A bytearray or array.array containing the mask bits.

    Returns:
        List[int]: A list of token IDs corresponding to bits set to 1 in the mask.
    """
    mask = bitarray(endian="big")  # Reads mask from left to right.
    mask.frombytes(mask_buffer.tobytes())
    return [i for i, bit in enumerate(mask) if bit]


def create_mask(size: int) -> array.array:
    """
    Creates a mask buffer initialized with zeros for a given number of bits.

    Args:
        size (int): The number of bits the mask should represent (e.g., vocab_size).

    Returns:
        array.array: A buffer of bytes initialized to zero, sized to hold `size` bits.
                     Each byte represents 8 bits, so the length is ceil(size / 8).

    Raises:
        ValueError: If size is not positive.
    """
    if size <= 0:
        raise ValueError("Mask size must be positive")
    byte_size = (size + 7) // 8
    return array.array("B", [0] * byte_size)


def first_token_id_from_mask(mask_buffer: array.array) -> int:
    bytes_data = mask_buffer.tobytes()

    # Parcourir chaque octet
    for byte_idx, byte in enumerate(bytes_data):
        if byte:  # Si l'octet contient au moins un bit à 1
            # Trouver le premier bit à 1 dans cet octet
            for bit_idx in range(8):
                if byte & (128 >> bit_idx):  # Vérifier le bit de gauche à droite (MSB)
                    return byte_idx * 8 + bit_idx

    return -1
