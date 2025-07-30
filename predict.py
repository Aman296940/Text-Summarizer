import os
import ssl
import nltk
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Attention, Concatenate, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Suppress TensorFlow warnings for cleaner output (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages

# Fix SSL context for nltk downloads if needed (avoids certificate issues)
try:
    _create_unverified_https = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https
except AttributeError:
    pass

# Function to download NLTK packages only if missing
def download_nltk_if_missing(packages):
    for package in packages:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)

nltk_packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
download_nltk_if_missing(nltk_packages)

# Parameters matching training setup
MAX_LEN_TEXT = 100
MAX_LEN_SUMM = 12
START_TOKEN = '<start>'
END_TOKEN = '<end>'

# Load tokenizers
with open('tokenizer_text.pkl', 'rb') as f:
    tokenizer_text = pickle.load(f)

with open('tokenizer_summ.pkl', 'rb') as f:
    tokenizer_sum = pickle.load(f)

# Load full model
model = load_model('text_summarizer_model')

# Build encoder model
enc_inputs = model.get_layer('encoder_inputs').input
enc_outputs, enc_h, enc_c = model.get_layer('encoder_lstm').output
enc_model = Model(enc_inputs, [enc_outputs, enc_h, enc_c])

# Build decoder model for inference with variable length input
dec_inputs = Input(shape=(None,), name='decoder_inputs')
enc_out_inf = Input(shape=(MAX_LEN_TEXT, 256), name='enc_out_inf')
state_h = Input(shape=(256,), name='state_h')
state_c = Input(shape=(256,), name='state_c')

dec_embed = model.get_layer('embedding_1')(dec_inputs)
dec_lstm = model.get_layer('decoder_lstm')
dec_out, h, c = dec_lstm(dec_embed, initial_state=[state_h, state_c])
attn_out = model.get_layer('attention_layer')([dec_out, enc_out_inf])
concat = Concatenate()([dec_out, attn_out])
dec_probs = model.get_layer('time_dist')(concat)

dec_model = Model([dec_inputs, enc_out_inf, state_h, state_c], [dec_probs, h, c])

# Text cleaning utilities using nltk
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# Summarization function
def summarize(text):
    cleaned = clean_text(text)
    seq = tokenizer_text.texts_to_sequences([cleaned])
    enc_out, h, c = enc_model.predict(pad_sequences(seq, maxlen=MAX_LEN_TEXT), verbose=0)
    summary_seq = [tokenizer_sum.word_index[START_TOKEN]]

    for _ in range(MAX_LEN_SUMM):
        dec_input_seq = pad_sequences([summary_seq], maxlen=None, padding='post')
        preds, h, c = dec_model.predict([dec_input_seq, enc_out, h, c], verbose=0)
        sampled_token = int(np.argmax(preds[0, -1, :]))
        if sampled_token == tokenizer_sum.word_index[END_TOKEN]:
            break
        summary_seq.append(sampled_token)

    idx2word = {v: k for k, v in tokenizer_sum.word_index.items()}
    summary_words = [idx2word.get(idx, '') for idx in summary_seq[1:]]
    return ' '.join(summary_words).strip()

# Command-line interface
if __name__ == "__main__":
    print("Text Summarizer Ready. Enter text to summarize (type 'exit' or 'quit' to stop):")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        if user_input:
            result = summarize(user_input)
            print("Summary:", result)
