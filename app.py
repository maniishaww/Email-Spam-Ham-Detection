import streamlit as st
import pandas as pd 
import numpy as np 
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# Loading the pre-trained model
model = pickle.load(open(r"C:\Users\Manisha\DATA SCIENCE\MACHINE LEARNING\nl processing\nb.pkl","rb"))

# Loading the CountVectorizer for training
with open(r"C:\Users\Manisha\DATA SCIENCE\MACHINE LEARNING\nl processing\bow.pkl","rb") as f:
    bow = pickle.load(f)

# Title of the page
    
st.title("Email Spam and Ham Detection")

#User input
mail = st.text_input("Enter the mail: ")

# Check if the mail is empty or not
if st.button('Submit'):
    if mail:
        data = bow.transform([mail]).toarray()
        result = model.predict(data)[0]
        st.write(result,"email!")
        if result == 'spam':
            st.image(r"C:\Users\Manisha\Downloads\spam image.png")