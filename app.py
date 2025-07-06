import streamlit as st
import pickle

# Load model and vectorizer using raw paths (Windows-safe)
with open(r'E:\Projects\Fake-News_Classifier\Model\fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'E:\Projects\Fake-News_Classifier\Model\tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit App
st.title("Fake News Classifier")
st.subheader("Enter a news article below and the model will predict if it's REAL or FAKE.")

user_input = st.text_area("News Article Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]
        label = "Real" if prediction == 1 else "Fake"
        st.success(f"Prediction: **{label}**")
