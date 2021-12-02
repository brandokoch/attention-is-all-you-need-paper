# Handle importing from src and models
import sys
import os
file_dir_pth = os.path.dirname(os.path.abspath(__file__))
models_dir_pth = os.path.join(os.path.dirname(file_dir_pth), 'models')
src_path=os.path.join(os.path.dirname(file_dir_pth),'src')
sys.path.append(src_path)

# Imports
from architectures.machine_translation_transformer import MachineTranslationTransformer
from tokenizers import Tokenizer
import streamlit as st
from glob import glob
import torch

# Configure streamlit layout
st.title("Transformer Inference App")

models=[os.path.basename(pth) for pth in glob(os.path.join(models_dir_pth,'*.pt'))]
tokenizers=[os.path.basename(pth) for pth in glob(os.path.join(models_dir_pth,'*.json'))]

selected_model=st.sidebar.selectbox("Model", models)
selected_tokenizer=st.sidebar.selectbox("Tokenizer", tokenizers)

text=st.text_input("Input text", "")
btn=st.button("Run model")

st.sidebar.subheader("Model Config")
d_model=st.sidebar.number_input("d_model", step=int(), value=512)
n_blocks=st.sidebar.number_input("n_blocks", step=int(), value=6)
n_heads=st.sidebar.number_input("n_heads", step=int(), value=8)
vocab_size=st.sidebar.number_input("vocab_size", step=int(), value=60000)
d_ff=st.sidebar.number_input("d_ff", step=int(), value=2048)
device=st.sidebar.radio("Device", ("CPU", "GPU"))


@st.cache
def load_models(model_pth):

    # Load the model from checkpoint
    model = MachineTranslationTransformer( 
        d_model=d_model,
        n_blocks=n_blocks,
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout_proba=0)
        
    if device=='GPU':
        model.load_state_dict(torch.load(model_pth,map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(model_pth,map_location=torch.device('cpu')))
    model.eval()

    return model


if btn:
    model_pth=os.path.join(models_dir_pth,selected_model)
    model=load_models(model_pth)

    tokenizer_pth=os.path.join(models_dir_pth,selected_tokenizer)
    tokenizer=Tokenizer.from_file(tokenizer_pth)

    out=model.translate(
        text, 
        tokenizer,
    )

    st.write(out)

