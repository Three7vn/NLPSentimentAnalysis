# NLPSentimentAnalysis
this project predicts the sentiment of a given message using a trained model and vectorizer created with scikit-learn. It classifies sentiment into three categories: hate speech, offensive language, and normal.

This project was trained on over 20,000 messages. However, this amount of data might not be sufficient for the most accurate model. Feel free to use different datasets with more data to produce a more accurate model. You can also improve the model by adding many different typo mapping functions, e.g. typo_mapping = {
    'fck': 'fuck',
    'fcker': 'fuck',
} and/or using the spellchecker library
To run locally, pip install scikit-learn joblib then run python sentiment_analysis.py :)

