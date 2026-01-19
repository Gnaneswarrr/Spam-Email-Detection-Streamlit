import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("Spam Email Detection App")
st.write("AI/ML Project using Streamlit")

# Dataset
data = {
    "message": [
        "Congratulations you won a lottery",
        "Free offer claim now",
        "Meeting scheduled at 10 AM",
        "Project discussion tomorrow",
        "Win cash prize today",
        "Please review the document"
    ],
    "label": ["spam", "spam", "ham", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'spam':1, 'ham':0})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# Train model
model = MultinomialNB()
model.fit(X, y)

# User input
user_input = st.text_input("Enter an email message:")

if st.button("Check Spam"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)

    if prediction[0] == 1:
        st.error("🚫 This is a SPAM Email")
    else:
        st.success("✅ This is NOT a Spam Email")
