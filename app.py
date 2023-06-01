import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

st.title("Admission Prediction")

answers = ["No", "Yes"]

gre = st.number_input("GRE Score")
toefl = st.number_input("TOEFL Score")
rating = st.number_input("University Rating")
sop = st.number_input("SOP")
lor = st.number_input("LOR")
cgpa = st.number_input("CGPA")
research = st.selectbox("Research", answers)

if st.button("Predict"):
	research = answers.index(research)
	test = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
	res = model.predict(test)
	print(res)
	st.success("Probability: " + str(res[0]))
