from collections import defaultdict
from random import shuffle
from typing import Dict, Optional

import nltk

ENGLISH_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def preprocess_text(text_origin: str, alphabet: Optional[str] = None) -> str:
    # make lower and delete punctuation
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text_origin.lower())

    # delete tokens with symbols not from the alphabet
    if alphabet is not None:
        tokens = [token for token in tokens if all(x in alphabet for x in token)]
    text_preprocessed = " ".join(tokens)
    return text_preprocessed


def encode_text(text_origin: str) -> str:
    text_alphabet = list(set(text_origin))

    shuffled_alphabet = text_alphabet[:]
    shuffle(shuffled_alphabet)

    encoding_key = {k: v for k, v in zip(text_alphabet, shuffled_alphabet)}
    text_encoded = ''.join(encoding_key.get(char, char) for char in text_origin)
    return text_encoded


def count_ngrams_frequencies(text: str, token_len: int) -> Dict[str, int]:
    assert len(text) > 0, "input text is empty"
    assert isinstance(token_len, int), "length of token should be a positive integer"
    assert token_len > 0, "length of token should be positive"

    freq_count = defaultdict(int)
    for i in range(len(text) - token_len + 1):
        ngram = text[i: i + token_len]
        freq_count[ngram] += 1
    return freq_count


def estimate_rate_of_matched_letters(text_true: str, text_guessed: str) -> float:
    assert len(text_true) == len(text_guessed)

    if len(text_true) == 0:
        return 0.0

    number_of_matched_letters = 0
    for letter_true, letter_guessed in zip(text_true, text_guessed):
        if letter_true == letter_guessed:
            number_of_matched_letters += 1
    return number_of_matched_letters / len(text_true)
