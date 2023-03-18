#PBL SEM IV - Diabetes prediction using Machine Learning.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv(r'C:\Users\kaust\.streamlit\diabetes-predicition.py')
st.title('Diabetes Prediction using ML')
st.subheader('Training Data Stats')
st.write(df.describe())
st.sidebar.subheader('Made By')
st.sidebar.write('Saahil Barve,Harshit Gala,Saravan Kota')
st.sidebar.title('Input your data by using the sliders below')

x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.title('Visualised Patient Report')

if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output ='You are not Diabetic'
  st.title(output)
  'Prevention is better than Cure\n'
  '1.Eat Healthy Food\n'
  '2.Quit Tobacoo\n'
  '3.Avoid unhealthy Fats\n'
  '4.Less alcohol Consumption'
else:
  output = 'You are Diabetic'
  st.title(output)
  '1.Lose extra weight\n'
  '2.Be more physically active\n'
  '3.Eat healthy plant foods\n' 
  '4.Eat healthy fats\n'
  '5.Skip fad diets and make healthier choices\n'     
  'Consult a nearby Doctor for further treatment\n'  

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')