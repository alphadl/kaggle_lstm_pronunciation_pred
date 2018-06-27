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

#----------#convert our entire dataset into two big 3D matrices (tensors)#-----------------#
##convert our entire dataset into two big 3D matricies(tensors):

MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns])
                         for _, pronuns in phonetic_dict.items()]
                        ) + 2  # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []

    for word, pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t, char in enumerate(word):
            word_matrix[t, :] = char_to_1_hot(char)
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t, :] = phone_to_1_hot(phone)

            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)

    return np.array(char_seqs), np.array(phone_seqs)


char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()
print('Word Matrix Shape: ', char_seq_matrix.shape)
print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)

"""
Word Matrix Shape:  (5343, 19, 30)
Pronunciation Matrix Shape:  (5343, 20, 87)
"""

phone_seq_matrix_decoder_output = np.pad(phone_seq_matrix,((0,0),(0,1),(0,0)), mode='constant')[:,1:,:]

from keras.models import Model
from keras.layers import Input, LSTM, Dense


def baseline_model(hidden_nodes=256):
    # Shared Components - Encoder
    char_inputs = Input(shape=(None, CHAR_TOKEN_COUNT))
    encoder = LSTM(hidden_nodes, return_state=True)

    # Shared Components - Decoder
    phone_inputs = Input(shape=(None, PHONE_TOKEN_COUNT))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True)
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')

    # Training Model
    _, state_h, state_c = encoder(char_inputs)  # notice encoder outputs are ignored
    encoder_states = [state_h, state_c]
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)

    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)

    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)

    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

    return training_model, testing_encoder_model, testing_decoder_model


from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

(char_input_train, char_input_test,
 phone_input_train, phone_input_test,
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output,
    test_size=TEST_SIZE, random_state=42)

TEST_EXAMPLE_COUNT = char_input_test.shape[0]

from keras.callbacks import ModelCheckpoint, EarlyStopping


def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
              batch_size=256,
              epochs=100,
              validation_split=0.2,  # Keras will automatically create a validation set for us
              callbacks=[checkpointer, stopper])

BASELINE_MODEL_WEIGHTS = os.path.join(
    './input','predict-english-pronunciations-model-weights','baseline_model_weights.hdf5'
)

training_model, testing_encoder_model, testing_decoder_model = baseline_model()
if not IS_KAGGLE:
    train(training_model, BASELINE_MODEL_WEIGHTS, char_input_train, phone_input_train, phone_output_train)



#----------#prediction#-----------------#
def predict_baseline(input_char_seq, encoder, decoder):
    state_vectors = encoder.predict(input_char_seq)

    prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
    prev_phone[0, 0, phone_to_id[START_PHONE_SYM]] = 1.

    end_found = False
    pronunciation = ''
    while not end_found:
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors)

        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]

        pronunciation += predicted_phone + ' '

        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN:
            end_found = True

        # Setup inputs for next time step
        prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]

    return pronunciation.strip()

# Helper method for converting vector representations back into words
def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


# Some words have multiple correct pronunciations
# If a prediction matches any correct pronunciation, consider it correct.
def is_correct(word,test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if test_pronunciation == correct_pronun:
            return True
    return False


def sample_baseline_predictions(sample_count, word_decoder):
    sample_indices = random.sample(range(TEST_EXAMPLE_COUNT), sample_count)
    for example_idx in sample_indices:
        example_char_seq = char_input_test[example_idx:example_idx+1]
        predicted_pronun = predict_baseline(example_char_seq, testing_encoder_model, testing_decoder_model)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word, predicted_pronun)
        print('✅ ' if pred_is_correct else '❌ ', example_word,'-->', predicted_pronun)

training_model.load_weights(BASELINE_MODEL_WEIGHTS)  # also loads weights for testing models
sample_baseline_predictions(10, one_hot_matrix_to_word)

"""
❌  DELOREAN --> D EH2 L ER0 IY1 N AH0
✅  POYNOR --> P OY1 N ER0
✅  STRAW --> S T R AO1
❌  PROCAINE --> P R OW0 K EY1 N IY0
✅  BATTLESHIP --> B AE1 T AH0 L SH IH2 P
❌  PRITHVI --> P R IH1 TH IY0
❌  GRUIS --> G R UW1 Z
✅  JANK --> JH AE1 NG K
✅  JON'S --> JH AA1 N Z
❌  MILISSENT --> M IH1 L IH0 S EH2 N T
"""


#----------#evaluation#-----------------#
def syllable_count(phonetic_sp):
    count = 0
    for phone in phonetic_sp.split():
        if phone[-1].isdigit():
            count += 1
    return count

# Examples:
for ex_word in list(phonetic_dict.keys())[:3]:
    print(ex_word, '--', syllable_count(phonetic_dict[ex_word][0]), 'syllables')


"""
SEVERT -- 2 syllables
BURLATSKY -- 3 syllables
UNIVA -- 3 syllables
"""

#----------#bleu score#-----------------#

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def is_syllable_count_correct(word, test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if syllable_count(test_pronunciation) == syllable_count(correct_pronun):
            return True
    return False


def bleu_score(word, test_pronunciation):
    references = [pronun.split() for pronun in phonetic_dict[word]]
    smooth = SmoothingFunction().method1
    return sentence_bleu(references, test_pronunciation.split(), smoothing_function=smooth)


def evaluate(test_examples, encoder, decoder, word_decoder, predictor):
    correct_syllable_counts = 0
    perfect_predictions = 0
    bleu_scores = []

    for example_idx in range(TEST_EXAMPLE_COUNT):
        example_char_seq = test_examples[example_idx:example_idx + 1]
        predicted_pronun = predictor(example_char_seq, encoder, decoder)
        example_word = word_decoder(example_char_seq)

        perfect_predictions += is_correct(example_word, predicted_pronun)
        correct_syllable_counts += is_syllable_count_correct(example_word, predicted_pronun)

        bleu = bleu_score(example_word, predicted_pronun)
        bleu_scores.append(bleu)

    syllable_acc = correct_syllable_counts / TEST_EXAMPLE_COUNT
    perfect_acc = perfect_predictions / TEST_EXAMPLE_COUNT
    avg_bleu_score = np.mean(bleu_scores)

    return syllable_acc, perfect_acc, avg_bleu_score


def print_results(model_name, syllable_acc, perfect_acc, avg_bleu_score):
    print(model_name)
    print('-' * 20)
    print('Syllable Accuracy: %s%%' % round(syllable_acc * 100, 1))
    print('Perfect Accuracy: %s%%' % round(perfect_acc * 100, 1))
    print('Bleu Score: %s' % round(avg_bleu_score, 4))


syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    char_input_test, testing_encoder_model, testing_decoder_model, one_hot_matrix_to_word, predict_baseline)
print_results('Baseline Model',syllable_acc, perfect_acc, avg_bleu_score)

"""
Baseline Model
--------------------
Syllable Accuracy: 92.7%
Perfect Accuracy: 61.3%
Bleu Score: 0.7138
"""