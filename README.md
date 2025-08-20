## 🔗 Download Pretrained Model

You can download the pretrained model here:  
👉 [text_summarizer_model.zip (GitHub Release)](https://github.com/Aman296940/Text-Summarizer/releases/download/v0.2.0-alpha/text_summarizer_model.zip)


# 📝 Text Summarizer (Seq2Seq + Attention)

This is a **text summarization project** built using **LSTM with Attention** in TensorFlow/Keras.  
It’s trained on the **Amazon Fine Food Reviews dataset** to generate short summaries of customer reviews.  

The project is designed to run smoothly even on a modest system:  
- **CPU:** AMD Ryzen 3  
- **RAM:** 8 GB  
- **Storage:** 512 GB SSD  

---

## 🌟 What it does
- Cleans and preprocesses text using **NLTK** (stopword removal, tokenization, lemmatization).  
- Creates simple extractive summaries (first 10 words).  
- Trains a **Seq2Seq + Attention model** to generate better summaries.  
- Works in both **training mode** and **interactive summarization mode**.  
- Saves the trained model and tokenizers for later use.  

---

## 📂 Project Files
.
├── text_summarizer.py # Main script (training + inference)
├── Reviews.csv # Dataset (Amazon Fine Food Reviews - download from Kaggle)
├── tokenizer_text.pkl # Saved tokenizer for review text
├── tokenizer_summ.pkl # Saved tokenizer for summaries
├── text_summarizer_model/ # Folder where trained model gets saved
└── README.md # This file :)

yaml
Copy
Edit

---

## ⚙️ Setup
Install the required libraries:
```bash
pip install tensorflow pandas numpy nltk
📊 Dataset
Dataset: Amazon Fine Food Reviews

Place the file Reviews.csv in the same folder as the script.

By default, only 50,000 rows are loaded to avoid memory issues.

▶️ How to Run
1. Train the model
bash
Copy
Edit
python text_summarizer.py
This will:
✔️ Load and clean the data
✔️ Train the Seq2Seq + Attention model
✔️ Save the model + tokenizers

2. Use it for summarization
After training, run the script again:

bash
Copy
Edit
python text_summarizer.py
You’ll get an interactive prompt:

vbnet
Copy
Edit
Text Summarizer Ready. Enter text (or 'exit'):
>> The food was absolutely wonderful, from preparation to presentation.
Summary: food absolutely wonderful preparation presentation
🔎 Example
Input:

css
Copy
Edit
I absolutely loved the pasta! The sauce was creamy and the flavor was authentic.
Output:

nginx
Copy
Edit
pasta sauce creamy flavor authentic
🧠 How it works
Encoder (LSTM) reads the input review.

Decoder (LSTM) generates the summary word by word.

Attention Layer helps the model focus on important parts of the text.

⚡ Optimizations for Low RAM
Loads only 50k reviews (instead of the full dataset).

Batch size reduced to 128.

Uses multi-threading (OMP settings) for CPU training.

📚 References
Amazon Fine Food Reviews Dataset

Seq2Seq Learning

Attention Mechanism
