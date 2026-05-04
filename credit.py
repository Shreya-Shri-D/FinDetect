import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Data preprocessing function
def preprocess_data(data):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test, scaler, X

# Logistic Regression Model class
class LogitRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        Z = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

# User input features function
def user_input_features(X):
    feature_dict = {}
    for col in X.columns:
        feature_dict[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    features = pd.DataFrame(feature_dict, index=[0])
    return features

# Main function for the Streamlit app
def credit():
    st.title("Credit Card Fraud Detection")
    st.write("""### Predict if a transaction is fraudulent based on user inputs.""")

    # Load and preprocess the data
    credit_card_data = load_data(os.path.join(_ROOT, "creditcard.csv"))
    X_train, X_test, Y_train, Y_test, scaler, X = preprocess_data(credit_card_data)

    # Train the model
    model = LogitRegression(learning_rate=0.001, iterations=90000)
    model.fit(X_train, Y_train)

    # User input for predictions
    input_data = user_input_features(X)
    scaled_data = scaler.transform(input_data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_data)
        if prediction[0] == 1:
            st.write("**Warning: Fraudulent Transaction**")
        else:
            st.write("**Transaction is Legitimate**")

    # Display model accuracy
    st.write("### Model Accuracy")
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    st.write(f'Accuracy on Training data: {training_data_accuracy * 100:.2f}%')

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    st.write(f'Accuracy on Test data: {test_data_accuracy * 100:.2f}%')

if __name__ == "__main__":
    credit()
