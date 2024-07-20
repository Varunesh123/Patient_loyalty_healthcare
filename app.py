import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

def load_data(file):
    df = pd.read_csv(file)
    return df
label_encoder = LabelEncoder()
def preprocess_data(df):
    df = df.fillna("0")
    cat_cols = df.describe(include='O').columns
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df, cat_cols

def train_model(df):
    df = df.drop("Loyalty Program Participation", axis=1)
    columns = df.columns
    columns = [col for col in columns if col not in ['Loyalty']]
    features = columns
    target = 'Loyalty'

    X = df[columns]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

    model = LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, average='macro')

    return model, accuracy, f1

st.title("Loyal Patients Identifier")

st.write("Upload your training CSV file:")
train_file = st.file_uploader("Choose a CSV file", type="csv")

if train_file is not None:
    df = load_data(train_file)
    st.write("Data Preview:")
    st.write(df.head())

    df, cat_cols = preprocess_data(df)

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, orient='h')
    plt.title('Box Plot of All Columns')
    st.pyplot(plt)

    model, accuracy, f1 = train_model(df)

    st.write(f"Model Accuracy: {accuracy}")
    st.write(f"Model F1 Score: {f1}")

    filename = "My_Model.sav"
    pickle.dump(model, open(filename, 'wb'))
    st.write("Model trained and saved!")

    st.write("Upload your test CSV file:")
    test_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
    if test_file is not None:
        test_df = load_data(test_file)
        test_df = test_df.fillna("0")

        for col in cat_cols:
            test_df[col] = label_encoder.fit_transform(test_df[col])

        loaded_model = pickle.load(open("My_Model.sav", 'rb'))
        test_df['Loyalty Prediction'] = loaded_model.predict(test_df)
        st.write("Predictions:")
        st.write(test_df)
