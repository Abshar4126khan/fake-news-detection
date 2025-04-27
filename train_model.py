# Additional training using feedback (pseudo-reinforcement)
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Load original dataset and retrain
# (Assuming X_train, y_train are your main training data)

# Load feedback examples
feedback_X = []
feedback_y = []

with open("feedback_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        text, label = line.strip().split("\t")
        feedback_X.append(text)
        feedback_y.append(1 if label == "Real" else 0)

# Vectorize
feedback_X_vec = vectorizer.transform(feedback_X)

# Partial training / updating
model.partial_fit(feedback_X_vec, feedback_y)

# Save the updated model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model updated with feedback!")
