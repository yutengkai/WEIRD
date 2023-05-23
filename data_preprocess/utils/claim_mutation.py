import nltk
from nltk.corpus import stopwords
from transformers import BartTokenizer, MarianMTModel, MarianTokenizer
import string
import random
import torch

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
remove_set = stop_words.union(punctuation)

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

def random_mask(token, rate=0.5):
    if random.random() < rate:
        return '<mask>'
    else:
        return token

def mask_tokens(sentence, rate=0.5):
    tokens = tokenizer.tokenize(sentence)
    tokens = [token.replace('Ġ', '') for token in tokens]
    masked_tokens = [random_mask(token, rate) if token not in remove_set else token
                     for token in tokens]
    return ' '.join(masked_tokens)

source_lang='en'
target_lang='zh'
BATCH_SIZE = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create the tokenizer and model
tokenizer_src = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')
model_src = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')

# Create the tokenizer and model for the reverse translation
tokenizer_tgt = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')
model_tgt = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')


def back_translate(sentence):

    # Translate to the target language
    translated = model_src.generate(**tokenizer_src.prepare_seq2seq_batch([sentence], return_tensors="pt"))
    translated_text = [tokenizer_src.decode(t, skip_special_tokens=True) for t in translated]

    # Translate back to the source language
    back_translated = model_tgt.generate(**tokenizer_tgt.prepare_seq2seq_batch(translated_text, return_tensors="pt"))
    back_translated_text = [tokenizer_tgt.decode(t, skip_special_tokens=True) for t in back_translated]
    
    return back_translated_text[0]

def translate_sentences(sentences, model, tokenizer, device='cuda'):
    model = model.to(device)
    translated_sentences = []

    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i+BATCH_SIZE]
        encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=32).to(device)
        with torch.no_grad():
            translated_batch = model.generate(**encoded_batch)
        for translated_sentence in translated_batch:
            translated_sentences.append(tokenizer.decode(translated_sentence))

    return translated_sentences