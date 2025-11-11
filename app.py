import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import streamlit as st
import re
import warnings
warnings.filterwarnings('ignore')


# # Deployment using Gradio


all_model=tf.keras.models.load_model('ner-AND-pos_model.h5',compile=False)
word2idx=joblib.load('word2idx.joblib')
tag2idx=joblib.load('tag2idx.joblib')
pos2idx=joblib.load('pos2idx.joblib')




def Predict_deploy(text):
    text=re.sub(r"([?.!,¿])", r" \1 ",text)
    
    max_len=50
    text_len=len(text.split())
    tokens=text.split()

        
    text=[[word2idx.get(w,0) for w in tokens]]
    text = pad_sequences(maxlen=max_len, sequences=text, padding="post", value=len(word2idx)-1)
    y_ner_pred,y_pos_pred =all_model.predict(np.array(text))
    # ----------------------------

    idx2pos={v:i for i,v in pos2idx.items()}
    idx2tag={v:i for i,v in tag2idx.items()}
    
    y_pos_pred=np.argmax(y_pos_pred,axis=-1)
    pred_poses=[idx2pos[i] for i in y_pos_pred[0]][:text_len]
    # ----------------------------
    y_ner_pred=np.argmax(y_ner_pred,axis=-1)
    pred_tags=[idx2tag[i] for i in y_ner_pred[0]][:text_len]
    # ----------------------------

    # Prepare dataframes

    NER_COLORS = {
    "B-per": "#aa9cfc",  # soft purple
    "I-per": "#cfc1ff",
    "B-org": "#ff9561",  # orange
    "I-org": "#ffc4a3",
    "B-geo": "#7aecec",  # teal
    "I-geo": "#aaf3f3",
    "B-gpe": "#bfeeb7",  # mint green
    "I-gpe": "#d6f5ce",
    "B-tim": "#feca74",  # yellow
    "I-tim": "#fee2a0",
    "B-art": "#e4e7d2",  # beige
    "I-art": "#f2f4e8",
    "B-eve": "#ffeb80",  # pale yellow
    "I-eve": "#fff3b0",
    "B-nat": "#e4c1f9",  # lavender
    "I-nat": "#edd3fa",
    "O": "#f0f0f0"
}

    
    pos_table = [[t, p] for t, p in zip(tokens, pred_poses)]
    ner_table = [[t, n] for t, n in zip(tokens, pred_tags)]
    # -------------------------

    # Create colored HTML for NER
    highlighted_text = ""
    for token, tag in zip(tokens, pred_tags):
        color = NER_COLORS.get(tag, "#DDDDDD")
        if tag != "O":
            highlighted_text += f"<span style='background-color:{color}; padding:3px; border-radius:4px; margin:2px'>{token} <sub><b>{tag}</b></sub></span> "
        else:
            highlighted_text += f"{token} "

    return highlighted_text.strip(), pos_table, ner_table
    
# Predict_deploy("My sister Sarah lives in Alexandria , Egypt .")


# pip install --upgrade gradio


# ------------------ Gradio Interface ------------------
# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="POS + NER Tagger", layout="wide")

st.title("🧠 POS + NER Tagger (Custom BiLSTM Model)")
st.markdown("Enter text to see both **Part-of-Speech (POS)** and **Named Entity Recognition (NER)** results.")

text_input = st.text_area("Input Sentence", height=100, placeholder="Example: My name is Mohammed Mahmoud I come from Egypt")

if st.button("🔍 Predict"):
    if text_input.strip():
        highlighted_text, pos_table, ner_table = Predict_deploy(text_input)
        
        st.subheader("Highlighted NER Text")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        st.subheader("Predicted POS Tags")
        st.dataframe(pos_table)
        
        st.subheader("Predicted NER Tags")
        st.dataframe(ner_table)
    else:
        st.warning("Please enter some text!")

# ------------------ Examples ------------------
st.markdown("---")
st.subheader("Try Example Sentences")
examples = [
    "My sister Sarah lives in Alexandria , Egypt .",
    "My name is Mohammed Mahmoud I come from Egypt . Contact me on mhmedmhmod184@gmail.com",
    "Cairo is the capital of Egypt .",
    "Ahmed works for Microsoft in Dubai ."

]

for ex in examples:
    if st.button(ex):
        text_input = ex
        highlighted_text, pos_table, ner_table = Predict_deploy(text_input)
        st.subheader("Highlighted NER Text")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        st.subheader("Predicted POS Tags")
        st.dataframe(pos_table)
        st.subheader("Predicted NER Tags")
        st.dataframe(ner_table)
