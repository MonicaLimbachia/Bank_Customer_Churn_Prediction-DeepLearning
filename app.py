import streamlit as st
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model, encoders and scaler file
model = tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app 
st.title('Customer Churn Prediction')

#User input 
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare Input data
input_data = pd.DataFrame({
    'Gender':gender,
    'Age':age,
    'Balance':balance,
    'Credit Score':credit_score,
    'Estimated Salary': estimated_salary,
    'Tenure':tenure,
    'Number of Products':num_of_products,
    'Has Credit Card':has_cr_card,
    'Is Active Member':is_active_member
},index=[0])

# One-hot encode "Geography"
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)
input_data.drop('Geography',axis=1,inplace=True)

input_data['Gender'] = label_encoder.transform(input_data['Gender'])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f'Churn Probsbility:{prediction_probability:.2f}')

if prediction_probability > 0.5:
    st.write('Customer is more likely to leave the bank.')
else:
    st.write('Customer is likely to stay with the bank.')