# chatapp/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
import nltk
import string
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure the required nltk data is downloaded
nltk.download('popular', quiet=True)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Lemmatizer and Stopwords
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess stop words to match the custom tokenizer's behavior
processed_stop_words = set([lemmer.lemmatize(word.lower()) for word in stop_words])

# Specify the file path
file_path = 'chatapp/history_pakistan.txt'  # relative path to the text file

# Read and preprocess the text file
def load_data(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        raw = f.read().lower()
    return raw

# Tokenization and Lemmatization
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in processed_stop_words]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Generate response
def response(user_response, sent_tokens):
    robot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robot_response = robot_response + "I think I need to read more about that..."
    else:
        robot_response = robot_response + sent_tokens[idx]
    sent_tokens.pop(-1)
    return robot_response

# Greetings
GREETING_INPUTS = ("assalamualaikum", "salam", "hello", "hi", "kia haal hai", "hey")
GREETING_RESPONSES = ["waalaikumsalam", "salam", "hello", "hi", "hey"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# View for the chatbot
def index(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_response = data.get('message', '').lower()
        raw_data = load_data(file_path)
        sent_tokens = nltk.sent_tokenize(raw_data)
        word_tokens = nltk.word_tokenize(raw_data)
        
        if user_response != 'bye!!':
            if user_response in ('thanks', 'thank you'):
                response_text = "EhsanBot: Anytime"
            else:
                greet_response = greeting(user_response)
                if greet_response is not None:
                    response_text = f"EhsanBot: {greet_response}"
                else:
                    response_text = f"EhsanBot: {response(user_response, sent_tokens)}"
        else:
            response_text = "EhsanBot: Take care.."
        
        return JsonResponse({"response": response_text})
    return render(request, 'chatapp/index.html')
