import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sys

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction[0]

test_cases = [
    "U.S. President Joe Biden visited the United Kingdom today to meet with the Prime Minister.",
    "Apple announced its new iPhone 15 today with a titanium body and an upgraded camera system.",
    "The stock market experienced a significant drop today due to concerns over inflation.",
    """WASHINGTON (Reuters) - The U.S. Senate voted on Thursday to confirm President Donald Trump’s nominee to lead the Department of Homeland Security, Kirstjen Nielsen, a former aide to White House chief of staff John Kelly. The Senate voted 62-37 to confirm Nielsen, a cybersecurity expert who previously served as Kelly’s deputy when he was homeland security secretary. She also served in the previous Republican administration of George W. Bush."""
]

print("Predictions:")
for t in test_cases:
    print("-" * 40)
    print("Text:", t[:60], "...")
    print("Result:", "Real" if fake_news_det(t) == 0 else "Fake")
