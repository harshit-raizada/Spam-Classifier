import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Applying custom CSS
st.markdown(
    """
    <style>
    /* Main page background color */
    body {
        background-color: #E8E8E8;
    }
    
    /* Text input box styling */
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #008CBA;
        color: white;
        background-color: #008CBA;
        padding: 10px 24px;
        cursor: pointer;
        text-align: center;
    }
    
    .stButton>button:hover {
        background-color: #005f73;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

# Sidebar for additional options or navigation
st.sidebar.title("About")
st.sidebar.info(
    "This application is designed to classify text messages as spam or not spam. "
    "Simply enter the text you want to classify and press the 'Predict' button."
)

# Streamlit app main body
st.title('Spam Classifier')

user_input = st.text_area("Enter Text", height=100)

if st.button('Predict'):
    # Preprocess user input
    transformed_text = transform(user_input)
    # Vectorize the user input
    vectorized_text = tfidf.transform([transformed_text]).toarray()
    # Make prediction
    prediction = model.predict(vectorized_text)
    # Display prediction
    if prediction == 0:
        st.success('The message is not spam. 🟢')
    else:
        st.error('The message is spam. 🔴')

# Adding a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #E8E8E8;
        color: grey;
        text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤️ by Harshit Raizada</p>
    </div>
    """,
    unsafe_allow_html=True
)