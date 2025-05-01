import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('trained_model.pkl', 'rb') )
st.title("Iris Prediction")
sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal width')

input_data = [sepal_length,sepal_width, petal_length,   petal_width]
input_data = np.asarray(input_data).reshape(1, -1)
if st.button('Result'):
    prediction = model.predict(input_data)
    st.success(prediction[0])
