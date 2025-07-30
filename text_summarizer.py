#!/usr/bin/env python3
# text_summarizer.py

"""
Text summarization with Seq2Seq LSTM + attention.
Optimized for AMD Ryzen 3 CPU, 8 GB RAM, 512 GB SSD.
Dataset: Amazon Fine Food Reviews ("Reviews.csv").
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import pickle

# 1 — ENVIRONMENT & THREADING OPTIMIZATION
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 2 — NLTK DATA DOWNLOAD (first run only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 3 — LOAD & PREPROCESS DATA
# Reduce rows to fit 8 GB RAM
NROWS = 50000
DATA_PATH = "Reviews.csv"
df = pd.read_csv(DATA_PATH, nrows=NROWS)

texts = df['Text'].astype(str).values

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(s):
    tokens = word_tokenize(s.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

cleaned_texts = [clean_text(s) for s in texts]

# Extractive summaries: first 10 words
summaries = [" ".join(t.split()[:10]) for t in cleaned_texts]

# 4 — TOKENIZATION & SEQUENCING
MAX_VOCAB = 50000
MAX_LEN_TEXT = 100
MAX_LEN_SUMM = 12

tokenizer_text = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer_text.fit_on_texts(cleaned_texts)
tokenizer_sum  = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer_sum.fit_on_texts(summaries)

seq_texts = tokenizer_text.texts_to_sequences(cleaned_texts)
seq_summ  = tokenizer_sum.texts_to_sequences(summaries)

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
tokenizer_sum.word_index[START_TOKEN] = len(tokenizer_sum.word_index) + 1
tokenizer_sum.word_index[END_TOKEN]   = len(tokenizer_sum.word_index) + 1

def add_tokens(seq_list):
    return [[tokenizer_sum.word_index[START_TOKEN]] + s + [tokenizer_sum.word_index[END_TOKEN]] for s in seq_list]

seq_summ = add_tokens(seq_summ)

encoder_input = pad_sequences(seq_texts, maxlen=MAX_LEN_TEXT, padding='post')
decoder_input = pad_sequences([s[:-1] for s in seq_summ], maxlen=MAX_LEN_SUMM, padding='post')
decoder_target= pad_sequences([s[1:]  for s in seq_summ], maxlen=MAX_LEN_SUMM, padding='post')

# 5 — MODEL DEFINITION
EMBED_DIM = 128
LATENT_DIM = 256
VOCAB_SIZE_TEXT = min(MAX_VOCAB, len(tokenizer_text.word_index) + 1)
VOCAB_SIZE_SUMM = min(MAX_VOCAB, len(tokenizer_sum.word_index)  + 1)

# Encoder
enc_inputs = Input(shape=(MAX_LEN_TEXT,), name='encoder_inputs')
enc_embed  = Embedding(VOCAB_SIZE_TEXT, EMBED_DIM, mask_zero=True)(enc_inputs)
encoder_lstm= LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm')
enc_outputs, enc_h, enc_c = encoder_lstm(enc_embed)

# Decoder
dec_inputs = Input(shape=(MAX_LEN_SUMM,), name='decoder_inputs')
dec_embed  = Embedding(VOCAB_SIZE_SUMM, EMBED_DIM, mask_zero=True)(dec_inputs)
decoder_lstm= LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
dec_outputs, _, _ = decoder_lstm(dec_embed, initial_state=[enc_h, enc_c])

# Attention
attn_layer = Attention(name='attention_layer')
attn_out   = attn_layer([dec_outputs, enc_outputs])
dec_concat = Concatenate(axis=-1)([dec_outputs, attn_out])

# Output
decoder_dense = TimeDistributed(Dense(VOCAB_SIZE_SUMM, activation='softmax'), name='time_dist')
decoder_outputs= decoder_dense(dec_concat)

model = Model([enc_inputs, dec_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 6 — TRAINING
# Reduced batch size to fit RAM
BATCH_SIZE = 128
EPOCHS     = 10

model.fit(
    [encoder_input, decoder_input],
    decoder_target[..., np.newaxis],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

# 7 — SAVE MODEL & TOKENIZERS
model.save('text_summarizer_model')
with open('tokenizer_text.pkl', 'wb') as f:
    pickle.dump(tokenizer_text, f)
with open('tokenizer_summ.pkl',  'wb') as f:
    pickle.dump(tokenizer_sum, f)

# 8 — INFERENCE SETUP (unchanged)
enc_model_inf = Model(enc_inputs, [enc_outputs, enc_h, enc_c])

dec_state_h  = Input(shape=(LATENT_DIM,))
dec_state_c  = Input(shape=(LATENT_DIM,))
enc_out_inf  = Input(shape=(MAX_LEN_TEXT, LATENT_DIM))

dec_embed2   = Embedding(VOCAB_SIZE_SUMM, EMBED_DIM, mask_zero=True)(dec_inputs)
dec_out2, state_h2, state_c2 = decoder_lstm(dec_embed2, initial_state=[dec_state_h, dec_state_c])
attn_out2    = attn_layer([dec_out2, enc_out_inf])
dec_inf_concat = Concatenate(axis=-1)([dec_out2, attn_out2])
dec_outputs2 = decoder_dense(dec_inf_concat)

dec_model_inf= Model(
    [dec_inputs, enc_out_inf, dec_state_h, dec_state_c],
    [dec_outputs2, state_h2, state_c2]
)

def summarize_text(input_text):
    cleaned = clean_text(input_text)
    seq     = tokenizer_text.texts_to_sequences([cleaned])
    enc_outs, h, c = enc_model_inf.predict(pad_sequences(seq, maxlen=MAX_LEN_TEXT), verbose=0)
    summary_seq = [tokenizer_sum.word_index[START_TOKEN]]
    for _ in range(MAX_LEN_SUMM):
        out_tokens, h, c = dec_model_inf.predict(
            [np.array(summary_seq)[None, :], enc_outs, h, c],
            verbose=0
        )
        sampled_token = np.argmax(out_tokens[0, -1, :])
        if sampled_token == tokenizer_sum.word_index[END_TOKEN]:
            break
        summary_seq.append(sampled_token)
    inv_map = {v: k for k, v in tokenizer_sum.word_index.items()}
    return " ".join(inv_map.get(idx, '') for idx in summary_seq[1:])

if __name__ == "__main__":
    print("Text Summarizer Ready. Enter text (or 'exit'):")
    while True:
        user_input = input(">> ")
        if user_input.lower().strip() in ('exit', 'quit'):
            break
        print("Summary:", summarize_text(user_input))
