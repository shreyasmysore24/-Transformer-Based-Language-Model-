import streamlit
import streamlit as st
import torch
from torch.nn import functional as F
from pt import GPTLanguageModel

# Load the GPTLanguageModel class from your file or define it here
# Make sure your 'bigram_language_model.pth' and 'wizard_of_oz.txt' files are in the same directory.

# Define the encode and decode functions
def encode(s, string_to_int):
    return [string_to_int[c] for c in s]

def decode(l, int_to_string):
    return ''.join([int_to_string[i] for i in l])

# Load model and setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load text data
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

# Load the state dictionary
state_dict = torch.load('bigram_language_model.pth', map_location=device)
model = GPTLanguageModel(vocab_size)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Streamlit UI
st.title("GPT Language Model Text Generator")
st.write("""
### About This Project
This project demonstrates a custom-trained GPT-like language model built using PyTorch. 
The model was trained on the text from *The Wizard of Oz* and uses Transformer-based 
architecture to generate text. The application allows users to input a prompt and 
generate new text based on it. 

Key highlights:
- Implemented a Transformer architecture with multi-head self-attention.
- Trained using PyTorch on character-level sequences.
- Allows dynamic interaction via Streamlit for real-time text generation.
""")

# User inputs
prompt = st.text_input("Enter a prompt:", value="Once upon a time")
text_length = st.number_input("Enter the number of tokens to generate:", min_value=10, max_value=5000, value=200)

# Generate text on button click
if st.button("Generate Text"):
    context = torch.tensor(encode(prompt, string_to_int), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(context.unsqueeze(0), max_new_tokens=text_length)
        generated_text = decode(generated[0].tolist(), int_to_string)
    st.subheader("Generated Text:")
    st.write(generated_text)
