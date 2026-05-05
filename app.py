import streamlit as st
import joblib

# 1. Load the saved model and vectorizer
model = joblib.load('svm_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 2. App Interface Construction
st.title("USM Classroom Facility Sentiment Classifier")


user_input = st.text_area("Enter student feedback here:")

if st.button("Analyze Sentiment"):
    if user_input:
        # 3. Process and Predict
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        
        # 4. Display Result
        if prediction == "Positive":
            st.success(f"Result: {prediction} 😊")
        elif prediction == "Negative":
            st.error(f"Result: {prediction} 😠")
        else:
            st.info(f"Result: {prediction} 😐")
    else:
        st.warning("Please enter some text first.")