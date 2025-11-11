import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import gradio as gr
import warnings
warnings.filterwarnings('ignore')


# # Deployment using Gradio


all_model=tf.keras.models.load_model('ner-AND-pos_model.h5')
word2idx=joblib.load('word2idx.joblib')
tag2idx=joblib.load('tag2idx.joblib')
pos2idx=joblib.load('pos2idx.joblib')




def Predict_deploy(text):
    
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

with gr.Blocks(theme="soft") as demo:
   gr.Markdown("## 🧠 POS + NER Tagger (Custom BiLSTM Model)")
   gr.Markdown("Enter text to see both <b>Part-of-Speech (POS)</b> and <b>Named Entity Recognition (NER)</b> results.")

   text_input = gr.Textbox(
       lines=3,
       label="Input Sentence",
       placeholder="Example: My name is Mohammed Mahmoud I come from Egypt"
   )

   ner_viz = gr.HTML(label="Highlighted NER Text")

   with gr.Row():
       pos_output = gr.Dataframe(headers=["Token", "POS Tag"], label="Predicted POS Tags", wrap=True)
       ner_output = gr.Dataframe(headers=["Token", "NER Tag"], label="Predicted NER Tags", wrap=True)

   submit_btn = gr.Button("🔍 Predict")
   submit_btn.click(Predict_deploy, inputs=text_input, outputs=[ner_viz, pos_output, ner_output])

   gr.Examples(
       examples=[
           ["My sister Sarah lives in Alexandria , Egypt ."],
           ["My name is Mohammed Mahmoud I come from Egypt . Contact me on mhmedmhmod184@gmail.com"],
           ["Cairo is the capital of Egypt ."],
           ["Ahmed works for Microsoft in Dubai ."]
       ],
       inputs=text_input
   )

# ------------------ Launch ------------------
if __name__ == "__main__":
   demo.launch()



