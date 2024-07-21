import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load the trained model and vectorizer
model = joblib.load('best_log_reg.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    text = re.sub(r'\n+', ' ', text)     # Remove newlines
    return text

# Function to predict sentiment
def predict_sentiment(text):
    clean_text_data = clean_text(text)
    text_vector = vectorizer.transform([clean_text_data])
    prediction = model.predict(text_vector)
    sentiment = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Normal'}
    result = sentiment[prediction[0]]
    return result

if __name__ == '__main__':
    while True:
        user_input = input("Enter a message to predict its sentiment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input)
        print(f'Predicted Sentiment: {sentiment}')
