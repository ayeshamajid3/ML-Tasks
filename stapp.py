import asyncio
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
model_name='berts_v123'
tokenizer=AutoTokenizer.from_pretrained(model_name)
classification=AutoModelForSequenceClassification.from_pretrained(model_name)

def rec(text):
    input=tokenizer(text,return_tensors='pt',truncation=True,padding=True)
    with torch.no_grad():
     output=classification(**input)
     probab=torch.nn.functional.softmax(output.logits,dim=-1)
     return probab.numpy()
        

    
import streamlit as st
st.title("Text Classification")
ui=st.text_area('enter your text:')

if st.button('classify'):
    pred=rec(ui)
    st.write(pred)