import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Select required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X = data['message']
y = data['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")