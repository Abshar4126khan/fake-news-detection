import streamlit as st
import pickle

# Load the trained model
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Function to predict
def predict_news(text):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    return "Real" if prediction[0] == 1 else "Fake"

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector (with Feedback)")
st.markdown("Enter a news article to check whether it's **Real** or **Fake**.")

# Input field
user_input = st.text_area("✍️ News Article", height=300)

if st.button("🔍 Check Now"):
    if user_input.strip():
        result = predict_news(user_input)
        st.subheader(f"🔎 Prediction: {'✅ REAL' if result == 'Real' else '❌ FAKE'}")

        # Save state
        st.session_state.prediction = result
        st.session_state.user_input = user_input
    else:
        st.warning("Please enter some news content.")

# Show feedback form
if "prediction" in st.session_state and st.session_state.prediction:
    st.markdown("---")
    st.subheader("🗳️ Feedback")
    feedback = st.radio("Was this prediction correct?", ("Yes", "No"))

    if st.button("Submit Feedback"):
        if feedback == "No":
            true_label = "Real" if st.session_state.prediction == "Fake" else "Fake"
            with open("feedback_data.txt", "a", encoding="utf-8") as f:
                f.write(f"{st.session_state.user_input}\t{true_label}\n")
            st.info("Thanks for your feedback! It will help improve the model.")
        else:
            st.success("Great! Glad the prediction was accurate. ✅")
