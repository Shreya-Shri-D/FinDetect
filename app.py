import streamlit as st
from Vishing import Vishing  # Import the function to display Vishing content
from qr import qr_page
from fake import fake_payment_detection
from credit import credit

# Set up session state to handle page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Display title and introduction
st.title("FinDetect")
st.write("Choose a detection type below:")

# Navbar using st.selectbox for page selection, always displayed at the top
page = st.selectbox(
    "Select Detection Type:",
    ("Home", "Vishing Detection", "Fake Detection", "Credit Card Fraud Detection", "QR Code Detection"),
    index=0  # Set default to 'Home'
)

# Update the session state based on navbar selection
if page == "Vishing Detection":
    st.session_state['page'] = 'vishing'
elif page == "Fake Detection":
    st.session_state['page'] = 'fake'
elif page == "Credit Card Fraud Detection":
    st.session_state['page'] = 'credit'
elif page == "QR Code Detection":
    st.session_state['page'] = 'qr'
else:
    st.session_state['page'] = 'home'

# Render content based on the selected page in session state
if st.session_state['page'] == 'vishing':
    Vishing()  # Display the Vishing Detection content

elif st.session_state['page'] == 'fake':
    fake_payment_detection()  # Display Fake Detection content

elif st.session_state['page'] == 'credit':
    credit()  # Display Credit Card Fraud Detection content

elif st.session_state['page'] == 'qr':
    qr_page()  # Display QR Code Detection content

else:
    st.write("Please select a detection type.")
