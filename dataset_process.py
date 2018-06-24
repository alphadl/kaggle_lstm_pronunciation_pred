# -*-coding:utf-8-*-
# Author: alphadl
# dataset_process.py 2018/6/24 22:18

#----------#dic construction#-----------------#
import re
import os
import random
import numpy as np

IS_KAGGLE = True

CMU_DICT_PATH = os.path.join('./input', 'cmudict-0.7b.txt')
CMU_SYMBOLS_PATH = os.path.join('./input', 'cmudict-0.7b.symbols.txt')

# skip words with numbers or symbols
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# setting a limit now simplifiers training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2


def load_clean_phonetic_dictionary():
    def is_alternate_pho_spelling(word):
        # no word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-2].isdigit() and word[-3] == '('

    def should_skip(word):
        if not word[0].isalpha():  # skip symbols
            return True
        if word[-1] == '.':  # skip abbreviations
            return True
        if re.search(ILLEGAL_CHAR_REGEX, word):
            return True
        if len(word) > MAX_DICT_WORD_LEN:
            return True
        if len(word) < MIN_DICT_WORD_LEN:
            return True
        return False

    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict:
        for line in cmu_dict:

            # skip commented lines
            if line[0:3] == ';;;':
                continue

            word, phonetic = line.strip().split('  ')

            # alternate pronounciations are formatted : "WORD(#) F AH0 N
            # EH1
            # we do not want to the "(#)" considered as part of the word

            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')]

            if should_skip(word):
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = []

            phonetic_dict[word].append(phonetic)

        if IS_KAGGLE:  # limit dataset to 5000 words
            phonetic_dict = {key: phonetic_dict[key]
                             for key in random.sample(list(phonetic_dict.keys()), 5000)}

        return phonetic_dict


phonetic_dict = load_clean_phonetic_dictionary()
example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])

'''
lets take a peek at our dictionary
'''

print("\n".join([k + ' --> ' + phonetic_dict[k][0] for k in random.sample(list(phonetic_dict.keys()), 10)]))
print('\nAfter cleaning, the dictionary contains %s words and %s pronunciations (%s are alternate pronunciations).' %
      (len(phonetic_dict), example_count, (example_count - len(phonetic_dict))))

"""
/Users/alphadl/.pyenv/versions/3.5.3/bin/python "/Users/alphadl/PycharmProjects/pronunciation prediction/dataset_process.py"
MILLSAPS --> M IH1 L S AE2 P S
DELUDES --> D IH0 L UW1 D Z
FOSIA --> F OW1 ZH AH0
OCEANSIDE --> OW1 SH AH0 N S AY2 D
KILDUFF --> K IH1 L D AH0 F
PULLEN --> P UH1 L AH0 N
TORBECK --> T AO1 R B EH0 K
SUNWORLD --> S AH1 N W ER2 L D
GATCH --> G AE1 CH
LYRA --> L AY1 R AH0

After cleaning, the dictionary contains 5000 words and 5371 pronunciations (371 are alternate pronunciations).
"""

#----------#data preparation#-----------------#


import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    with open(CMU_SYMBOLS_PATH) as file:
        for line in file:
            phone_list.append(line.strip())
    return [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str

#create character to ID mapping
char_to_id,id_to_char = id_mappings_from_list(char_list())

#load phonetic symbols and create ID mappings
phone_to_id,id_to_phone = id_mappings_from_list(phone_list())

#example
print('char to id mapping:\n', char_to_id)

"""
char to id mapping:
 {'': 0, 'P': 19, 'A': 4, 'F': 9, 'T': 23, 'X': 27, 'C': 6, 'E': 8, 'K': 14, 'D': 7, 'H': 11, 'W': 26, 'G': 10, 'B': 5, 'O': 18, 'U': 24, 'I': 12, 'Y': 28, 'M': 16, '.': 1, 'Z': 29, 'S': 22, 'N': 17, 'R': 21, 'L': 15, "'": 3, 'Q': 20, '-': 2, 'V': 25, 'J': 13}

Process finished with exit code 0
"""

#----------#1-hot vectors construction#-----------------#

CHAR_TOKEN_COUNT = len(char_to_id)
PHONE_TOKEN_COUNT = len(phone_to_id)

def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec

def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec

#example
print('"A" is represented by:\n',char_to_1_hot('A'),'\n------')
print('"AH0" is represented by:\n',phone_to_1_hot('AH0'))

"""
"A" is represented by:
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.] 
------
"AH0" is represented by:
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
"""