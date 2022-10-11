
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import time





st.header("Cost Model")
opt = st.sidebar.radio(label='Result',options=['r2Score','prediction'])

if opt=='r2Score':
    df = pd.read_excel('Retention_CP_prediction.xlsx')
    encoder = OrdinalEncoder()
    final_data = encoder.fit_transform(df.drop(columns='cost'))
    X = final_data
    y = df['cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    t1 = time.time() - t0
    st.write('r2 score: ', r2_score(y_test, y_pred))
    st.write('time : ', t1)
    

elif opt=='prediction':
    df = pd.read_excel('Retention_CP_prediction.xlsx')
    encoder = OrdinalEncoder()
    final_data = encoder.fit_transform(df.drop(columns='cost'))
    X = final_data
    y = df['cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model = DecisionTreeRegressor(random_state=44)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    st.table(predictions)

else:
    st.write('Please type input') 