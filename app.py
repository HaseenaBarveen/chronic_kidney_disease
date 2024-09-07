#importing 
import pandas as pd
import streamlit as st
import base64

#adding background images
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string=base64.b64encode(image.read())
    st.markdown(
        f"""
        <style>
        .stApp{{
                background-image:url(data:image/{"jpg"};base64,{encoded_string.decode()});
                background-size:cover
                }}
        </style>
        """,
        unsafe_allow_html=True
                 )
add_bg_from_local("download.jpg")

#reading the file   
data=pd.read_csv("kidney_disease.csv")
data.isnull()
data.isnull().sum()

#dropping null values
data.dropna(inplace=True)
data.isnull().sum()

#label encoding
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data["classification"]=lab.fit_transform(data["classification"])
data=data.astype(str).apply(lab.fit_transform)
print(data)

x=data[["age","bp","sg","rbc"]]
print(x)
y=data["classification"]
print(y)

#splitting for test and train
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)  

#decision tree
from sklearn.tree import DecisionTreeClassifier
ypred=DecisionTreeClassifier()
ypred.fit(xtrain,ytrain)
pred=ypred.predict(xtest)

#random classifier
from sklearn.ensemble import RandomForestClassifier
Re=RandomForestClassifier()
Re.fit(xtrain,ytrain)
pred1=Re.predict(xtest)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

acc1 = accuracy_score(ytest,pred)
print(acc1)

acc2 = accuracy_score(ytest,pred1)

print(confusion_matrix(ytest,pred))


print(classification_report(ytest,pred))

print(xtest.iloc[0])
import numpy as np

var=xtest.iloc[28]
var=np.array(var)
var1=var.reshape(1,-1)
pred_f=Re.predict(var1)
print(pred_f)

import matplotlib.pyplot as plt 

items=[acc1,acc2]
height=range(len(items))
labels=['DTC','RF']
plt.bar(height,items,tick_label=labels,width=0.5,color=["orange","blue"])
plt.xlabel("methods")
plt.ylabel("height")
plt.show()

#app
st.title("Predicting kidney disease")
st.text("Enter the details")
name=st.text_input("Enter your name")
age=st.number_input("Enter your age")
bp=st.number_input("Enter your blood pressure")
sg=st.number_input("enter your sugar level")
rbc=st.text_input("enter your rbc is normal/abnormal")

if st.button("Result"):


    arr=np.array([age,bp,sg,rbc])

    ar=arr.reshape(1,-1)
    pr=Re.predict(ar)
    st.write("Hello",name)
    
    
    if pr==0:
        st.write("Chronic kidney disease")
    else:
        st.write("not a chronic kidney disease")