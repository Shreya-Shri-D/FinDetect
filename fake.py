import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))

def fake_payment_detection():
    # Load the dataset
    data = pd.read_csv(os.path.join(_ROOT, "fake_payments_dataset.csv"))

    # Sidebar title
    st.sidebar.title('User Input')

    # Input fields for user data
    transaction_amount = st.sidebar.number_input(
        'Transaction Amount',
        min_value=data['Transaction_Amount'].min(),
        max_value=data['Transaction_Amount'].max(),
        value=data['Transaction_Amount'].mean()
    )
    transaction_frequency = st.sidebar.slider(
        'Transaction Frequency',
        min_value=int(data['Transaction_Frequency'].min()),
        max_value=int(data['Transaction_Frequency'].max()),
        value=int(data['Transaction_Frequency'].mean())
    )
    merchant_rating = st.sidebar.slider(
        'Merchant Rating',
        min_value=int(data['Merchant_Rating'].min()),
        max_value=int(data['Merchant_Rating'].max()),
        value=int(data['Merchant_Rating'].mean())
    )
    customer_age = st.sidebar.slider(
        'Customer Age',
        min_value=int(data['Customer_Age'].min()),
        max_value=int(data['Customer_Age'].max()),
        value=int(data['Customer_Age'].mean())
    )
    transaction_location_international = st.sidebar.checkbox('Transaction Location International')
    transaction_location_local = st.sidebar.checkbox('Transaction Location Local')

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Transaction_Amount': [transaction_amount],
        'Transaction_Frequency': [transaction_frequency],
        'Merchant_Rating': [merchant_rating],
        'Customer_Age': [customer_age],
        'Transaction_Location_International': [transaction_location_international],
        'Transaction_Location_Local': [transaction_location_local]
    })

    # Split features and target
    X = data.drop('Receipt_Details', axis=1)
    y = data['Receipt_Details']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Support Vector Machine (SVM) model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Make prediction for user input
    prediction = svm_model.predict(input_data)
    prediction_proba = svm_model.predict_proba(input_data)[:, 1]

    # Display prediction result
    st.title('Fake Payment Detection')
    if prediction[0] == 1:
        st.write('Prediction: Fake Payment')
    else:
        st.write('Prediction: Genuine Payment')

    # Calculate probabilities on the test set for ROC curve
    y_test_proba = svm_model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Main function to control the flow of the app
def main():
    fake_payment_detection()

if __name__ == "__main__":
    main()
