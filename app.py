import argparse
import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import requests
from bs4 import BeautifulSoup
import easyocr
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from functools import lru_cache
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Handle Missing Dependencies ---
MISSING_DEPENDENCIES = []

try:
    import mysql.connector
except ImportError:
    MISSING_DEPENDENCIES.append("mysql-connector-python")
try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING_DEPENDENCIES.append("beautifulsoup4")
try:
    import easyocr
except ImportError:
    MISSING_DEPENDENCIES.append("easyocr")
try:
    import cv2
except ImportError:
    MISSING_DEPENDENCIES.append("opencv-python")

if MISSING_DEPENDENCIES:
    print(f"\n❌ Missing dependencies: {', '.join(MISSING_DEPENDENCIES)}")
    print("To install, run: pip install " + " ".join(MISSING_DEPENDENCIES))
    exit(1)

# --- Create Directories ---
MODELS_DIR = "models"
UPLOAD_FOLDER = "uploads"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- DB Connection (Optional) ---
cursor = None
db = None

def setup_database():
    global db, cursor
    try:
        db = mysql.connector.connect(host="localhost", user="root", password="qwerty00", database="fake_news_project")
        cursor = db.cursor()
        print("✅ Database connection established.")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Will proceed without database features. Using CSV data only.")
        return False

# --- API Key ---
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def is_url(text):
    url_pattern = r'^(https?:\/\/)?([\w\-]+(\.[\w\-]+)+[/#?]?.*)$'
    return bool(re.match(url_pattern, text.strip()))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- OPTIMIZED OCR Implementation ---
class OCREngine:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("🔄 Initializing EasyOCR reader...")
            cls._instance = easyocr.Reader(
                ['en'], 
                gpu=False, 
                quantize=True, 
                verbose=False,
                model_storage_directory=None,
                download_enabled=True
            )
            print("✅ EasyOCR initialized")
        return cls._instance

@lru_cache(maxsize=32)
def extract_text_from_image(image_path, min_confidence=0.2):
    try:
        start_time = time.time()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = img.shape[:2]
        max_dim = 1800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        ocr = OCREngine.get_instance()
        result = ocr.readtext(gray, detail=1, paragraph=True, min_size=10)
        
        text_blocks = []
        for detection in result:
            if len(detection) == 3:
                bbox, text, confidence = detection
                if confidence > min_confidence:
                    text_blocks.append(text)
            elif len(detection) == 2:
                bbox, text = detection
                text_blocks.append(text)
                print(f"⚠️ No confidence score for text: {text[:50]}...")
            else:
                print(f"⚠️ Unexpected detection format: {detection}")
                continue
        
        text = ' '.join(text_blocks)
        text = re.sub(r'\s+', ' ', text).strip()
        
        processing_time = time.time() - start_time
        if text:
            print(f"✅ Extracted {len(text)} characters in {processing_time:.2f} seconds")
            print(f"   First 50 chars: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            print(f"⚠️ No text detected in image after {processing_time:.2f} seconds")
            
        return text
    
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {str(e)}")
        return ""

# --- URL Processing ---
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        article = soup.find('article') or soup.find('div', class_=re.compile('content|article|post', re.I))
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            raise ValueError("No meaningful text extracted from URL.")
        print(f"✅ Extracted {len(text)} characters from URL: {url}")
        return text
    except Exception as e:
        print(f"❌ Error fetching or parsing URL {url}: {e}")
        return None

# --- Model Training ---
def train_models(include_groq_feedback=False, groq_text=None, groq_label=None, extra_weight=1):
    additional_real = []
    additional_fake = []
    
    try:
        with open(os.path.join(MODELS_DIR, 'additional_examples.pkl'), 'rb') as f:
            additional_examples = pickle.load(f)
            additional_real = additional_examples.get('real', [])
            additional_fake = additional_examples.get('fake', [])
            print(f"✅ Loaded {len(additional_real)} additional real news and {len(additional_fake)} additional fake news examples")
    except Exception as e:
        print(f"ℹ️ No additional examples found or error loading: {e}")
    
    real_news = []
    if cursor:
        try:
            cursor.execute("SELECT title, description, content FROM news_articles")
            records = cursor.fetchall()
            real_news = [' '.join(filter(None, r)) for r in records if any(r)]
            print(f"✅ Loaded {len(real_news)} real news articles from database")
        except Exception as e:
            print(f"❌ Error loading from database: {e}")
    
    if not real_news:
        print("ℹ️ Using alternative real news data source...")
        try:
            true_df = pd.read_csv("True.csv")
            if not true_df.empty:
                real_news = true_df['title'].dropna().tolist()
                print(f"✅ Loaded {len(real_news)} real news articles from True.csv")
        except Exception as e:
            print(f"❌ Error loading True.csv: {e}")
            if not real_news:
                print("⚠️ No real news data found. Creating dummy data for demo purposes.")
                real_news = [
                    "NASA Confirms Evidence of Water on Mars",
                    "Scientists Develop New Cancer Treatment",
                    "Federal Reserve Announces Interest Rate Decision",
                    "New Study Links Exercise to Longevity",
                    "Stock Market Closes at Record High"
                ]

    fake_news = []
    try:
        fake_df = pd.read_csv("Fake.csv")
        fake_news = fake_df['title'].dropna().tolist()
        print(f"✅ Loaded {len(fake_news)} fake news articles from Fake.csv")
    except Exception as e:
        print(f"❌ Error loading Fake.csv: {e}")
        print("⚠️ No fake news data found. Creating dummy data for demo purposes.")
        fake_news = [
            "Aliens Make Contact With Government Officials",
            "Miracle Cure Discovered That Big Pharma Is Hiding",
            "Scientists Confirm the Earth is Actually Flat",
            "5G Networks Secretly Controlling Minds",
            "Celebrity Secretly Replaced by Clone"
        ]

    real_news.extend(additional_real)
    fake_news.extend(additional_fake)
    
    if include_groq_feedback and groq_text and groq_label is not None:
        if groq_label == 1:
            additional_real.append(groq_text)
            for _ in range(extra_weight):
                real_news.append(groq_text)
        else:
            additional_fake.append(groq_text)
            for _ in range(extra_weight):
                fake_news.append(groq_text)
        print(f"✅ Added Groq feedback as {'real' if groq_label == 1 else 'fake'} news with weight {extra_weight}")
        
        try:
            with open(os.path.join(MODELS_DIR, 'additional_examples.pkl'), 'wb') as f:
                pickle.dump({'real': additional_real, 'fake': additional_fake}, f)
            print(f"✅ Saved {len(additional_real)} real and {len(additional_fake)} fake additional examples")
        except Exception as e:
            print(f"❌ Error saving additional examples: {e}")

    min_len = min(len(real_news), len(fake_news))
    real_news = real_news[:min_len]
    fake_news = fake_news[:min_len]
    
    print(f"ℹ️ Training with {len(real_news)} real and {len(fake_news)} fake news articles")

    texts = real_news + fake_news
    labels = [1]*len(real_news) + [0]*len(fake_news)
    texts = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    print("\n🔄 Training SGD Classifier model...")
    start_time = time.time()
    sgd_model = SGDClassifier(
        loss='hinge', 
        penalty='l2',
        alpha=1e-3, 
        random_state=42,
        max_iter=100,
        tol=1e-3,
        class_weight='balanced'
    )
    sgd_model.fit(X_train, y_train)
    sgd_training_time = time.time() - start_time
    
    sgd_preds = sgd_model.predict(X_test)
    print(f"⏱️ SGD Classifier training completed in {sgd_training_time:.2f} seconds")
    print("📊 SGD Classifier Results:")
    print(classification_report(y_test, sgd_preds))
    
    if hasattr(sgd_model, 'coef_'):
        sgd_feature_names = vectorizer.get_feature_names_out()
        sgd_feature_importance = sorted(zip(sgd_feature_names, abs(sgd_model.coef_[0])), 
                                   key=lambda x: x[1], reverse=True)
        print("\n🔍 Top 10 Most Important Features (SGD Classifier):")
        for feature, importance in sgd_feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")

    print("\n🚀 Training Passive Aggressive Classifier model...")
    start_time = time.time()
    pa_model = PassiveAggressiveClassifier(
        C=1.0,
        random_state=42,
        max_iter=100,
        tol=1e-3,
        class_weight='balanced'
    )
    pa_model.fit(X_train, y_train)
    pa_training_time = time.time() - start_time
    
    pa_preds = pa_model.predict(X_test)
    print(f"⏱️ Passive Aggressive Classifier training completed in {pa_training_time:.2f} seconds")
    print("📊 Passive Aggressive Classifier Results:")
    print(classification_report(y_test, pa_preds))

    if hasattr(pa_model, 'coef_'):
        pa_feature_names = vectorizer.get_feature_names_out()
        pa_feature_importance = sorted(zip(pa_feature_names, abs(pa_model.coef_[0])), 
                                  key=lambda x: x[1], reverse=True)
        print("\n🔍 Top 10 Most Important Features (Passive Aggressive Classifier):")
        for feature, importance in pa_feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
    
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if include_groq_feedback else ""
    try:
        with open(os.path.join(MODELS_DIR, f'vectorizer{timestamp}.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(MODELS_DIR, f'sgd_model{timestamp}.pkl'), 'wb') as f:
            pickle.dump(sgd_model, f)
        with open(os.path.join(MODELS_DIR, f'pa_model{timestamp}.pkl'), 'wb') as f:
            pickle.dump(pa_model, f)
        print(f"✅ Models saved to {MODELS_DIR}{' with timestamp: ' + timestamp if timestamp else ''}")
    except Exception as e:
        print(f"❌ Error saving models: {e}")

    return sgd_model, pa_model, vectorizer

# --- Load Models ---
def load_models():
    try:
        with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'sgd_model.pkl'), 'rb') as f:
            sgd_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'pa_model.pkl'), 'rb') as f:
            pa_model = pickle.load(f)
        print("✅ Models loaded successfully from models directory.")
        return sgd_model, pa_model, vectorizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("🔄 Training new models instead...")
        return train_models()

# --- Prediction Functions ---
def get_model_predictions(sgd_model, pa_model, vectorizer, news_text):
    news_text = clean_text(news_text)
    vectorized = vectorizer.transform([news_text])
    
    sgd_prediction = sgd_model.predict(vectorized)[0]
    sgd_decision = sgd_model.decision_function(vectorized)[0]
    sgd_confidence = 1 / (1 + np.exp(-abs(sgd_decision)))
    sgd_result = "Real News" if sgd_prediction == 1 else "Fake News"
    
    pa_prediction = pa_model.predict(vectorized)[0]
    pa_decision = pa_model.decision_function(vectorized)[0]
    pa_confidence = 1 / (1 + np.exp(-abs(pa_decision)))
    pa_result = "Real News" if pa_prediction == 1 else "Fake News"
    
    return sgd_result, sgd_confidence, pa_result, pa_confidence

# --- AI Functions ---
def call_groq_api(messages, temperature=0.2, max_tokens=512):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "stream": False
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ API response is empty or invalid."
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return None

def analyze_with_ai(news_text, is_followup=False, user_question=None):
    print("🔄 Making API call to Groq...")
    system_content = "You are a helpful assistant that evaluates whether news is real or fake."
    if is_followup:
        system_content += " You have expertise in journalistic standards, fact-checking, and media literacy."
        user_content = f"Regarding this news: '{news_text}'\n\nUser question: {user_question}"
    else:
        user_content = f"Is the following news real or fake? Please explain your reasoning: '{news_text}'"
        
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    response = call_groq_api(
        messages=messages,
        temperature=0.2 if not is_followup else 0.3,
        max_tokens=512 if not is_followup else 1024
    )
    
    if response:
        return response
    else:
        print("⚠️ Failed to get response from Groq API")
        return perform_rule_based_analysis(news_text)

def perform_rule_based_analysis(news_text):
    news_lower = news_text.lower()
    sensationalist_words = ["shocking", "incredible", "unbelievable", "mind-blowing", 
                          "you won't believe", "secret", "conspiracy", "miracle", 
                          "amazing", "stunning", "jaw-dropping"]
    sensationalist_count = sum(1 for word in sensationalist_words if word in news_lower)
    excessive_punctuation = len(re.findall(r'[!?]{2,}', news_text)) > 0
    all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', news_text))
    has_clickbait = "click here " in news_lower or "you won't believe" in news_lower
    has_urgency = "act now" in news_lower or "limited time" in news_lower
    
    total_red_flags = sensationalist_count + excessive_punctuation + all_caps_words + has_clickbait + has_urgency
    
    analysis = [
        "Based on text analysis (without AI):",
        f"- Sensationalist language: {sensationalist_count} instances",
        f"- Excessive punctuation: {'Yes' if excessive_punctuation else 'No'}",
        f"- ALL CAPS words: {all_caps_words} instances",
        f"- Clickbait phrases: {'Present' if has_clickbait else 'None detected'}",
        f"- Urgency tactics: {'Present' if has_urgency else 'None detected'}"
    ]
    
    verdict = "Likely FAKE NEWS" if total_red_flags > 2 else "Possibly REAL NEWS"
    analysis.append(f"\nVerdict: {verdict}")
    
    return "\n".join(analysis)

def check_ai_agreement(ai_response, sgd_prediction, pa_prediction):
    response_lower = ai_response.lower()
    if "real news" in response_lower or "news is real" in response_lower:
        ai_verdict = "Real News"
    elif "fake news" in response_lower or "news is fake" in response_lower:
        ai_verdict = "Fake News"
    else:
        real_indicators = ["credible", "legitimate", "trustworthy", "authentic", "factual"]
        fake_indicators = ["false", "misleading", "misinformation", "fabricated", "unreliable"]
        real_count = sum(1 for word in real_indicators if word in response_lower)
        fake_count = sum(1 for word in fake_indicators if word in response_lower)
        if real_count > fake_count:
            ai_verdict = "Real News"
        elif fake_count > real_count:
            ai_verdict = "Fake News"
        else:
            return False, "Uncertain"
    
    if ai_verdict == sgd_prediction or ai_verdict == pa_prediction:
        print(f"✅ AI ({ai_verdict}) agrees with model prediction.")
        return True, ai_verdict
    else:
        print(f"❌ AI ({ai_verdict}) disagrees with model predictions ({sgd_prediction}, {pa_prediction}).")
        return False, ai_verdict

def process_news(news_text, sgd_model, pa_model, vectorizer, output_widget):
    output_widget.insert(tk.END, f"\n📰 Analyzing: {news_text}\n")
    output_widget.see(tk.END)
    
    start_time = time.time()
    sgd_pred, sgd_conf, pa_pred, pa_conf = get_model_predictions(
        sgd_model, pa_model, vectorizer, news_text
    )
    prediction_time = time.time() - start_time
    
    output_widget.insert(tk.END, f"🔄 SGD Classifier: {sgd_pred} (Confidence: {sgd_conf*100:.2f}%)\n")
    output_widget.insert(tk.END, f"🚀 Passive Aggressive: {pa_pred} (Confidence: {pa_conf*100:.2f}%)\n")
    output_widget.insert(tk.END, f"⏱️ Prediction completed in {prediction_time:.4f} seconds\n")
    output_widget.see(tk.END)
    
    def handle_ai_cross_check():
        output_widget.insert(tk.END, "\n🧠 Cross-checking with AI...\n")
        output_widget.see(tk.END)
        ai_check = analyze_with_ai(news_text)
        output_widget.insert(tk.END, f"🔄 AI Analysis:\n{ai_check}\n")
        output_widget.see(tk.END)
        
        agreementyside, ai_verdict = check_ai_agreement(ai_check, sgd_pred, pa_pred)
        
        if not agreement:
            if messagebox.askyesno("Retrain Models", "AI disagrees with model predictions.\nWould you like to retrain with AI feedback?"):
                groq_label = 1 if ai_verdict == "Real News" else 0
                output_widget.insert(tk.END, "\n🧠 Retraining models with AI feedback...\n")
                output_widget.see(tk.END)
                new_sgd_model, new_pa_model, new_vectorizer = train_models(
                    include_groq_feedback=True, 
                    groq_text=news_text, 
                    groq_label=groq_label
                )
                
                new_sgd_pred, new_sgd_conf, new_pa_pred, new_pa_conf = get_model_predictions(
                    new_sgd_model, new_pa_model, new_vectorizer, news_text
                )
                output_widget.insert(tk.END, "\n📊 Retrained Model Predictions:\n")
                output_widget.insert(tk.END, f"🔄 SGD Classifier: {new_sgd_pred} (Confidence: {new_sgd_conf*100:.2f}%)\n")
                output_widget.insert(tk.END, f"🚀 Passive Aggressive: {new_pa_pred} (Confidence: {new_pa_conf*100:.2f}%)\n")
                output_widget.see(tk.END)
                return new_sgd_model, new_pa_model, new_vectorizer
        return sgd_model, pa_model, vectorizer

    if pa_pred == "Fake News" and sgd_pred == "Real News":
        output_widget.insert(tk.END, "\n⚠️ Model disagreement detected: PA says Fake News but SGD says Real News\n")
        output_widget.insert(tk.END, "Automatically cross-checking with AI...\n")
        output_widget.see(tk.END)
        ai_check = analyze_with_ai(news_text)
        output_widget.insert(tk.END, f"🔄 AI Analysis:\n{ai_check}\n")
        output_widget.see(tk.END)
        
        agreement, ai_verdict = check_ai_agreement(ai_check, sgd_pred, pa_pred)
        
        if ai_verdict == "Fake News":
            output_widget.insert(tk.END, "\n✅ AI confirms Passive Aggressive classification: This is likely Fake News\n")
            output_widget.see(tk.END)
            if messagebox.askyesno("Retrain Models", "Would you like to retrain models with this example as Fake News?"):
                sgd_model, pa_model, vectorizer = train_models(
                    include_groq_feedback=True, 
                    groq_text=news_text, 
                    groq_label=0,
                    extra_weight=10
                )
                
                sgd_pred, sgd_conf, pa_pred, pa_conf = get_model_predictions(
                    sgd_model, pa_model, vectorizer, news_text
                )
                output_widget.insert(tk.END, "\n📊 Retrained Model Predictions:\n")
                output_widget.insert(tk.END, f"🔄 SGD Classifier: {sgd_pred} (Confidence: {sgd_conf*100:.2f}%)\n")
                output_widget.insert(tk.END, f"🚀 Passive Aggressive: {pa_pred} (Confidence: {pa_conf*100:.2f}%)\n")
                output_widget.see(tk.END)
                
                if sgd_pred == "Real News" and messagebox.askyesno("Stronger Training", "SGD model still predicts Real News. Apply stronger training?"):
                    sgd_model, pa_model, vectorizer = train_models(
                        include_groq_feedback=True, 
                        groq_text=news_text, 
                        groq_label=0,
                        extra_weight=50
                    )
                    
                    sgd_pred, sgd_conf, pa_pred, pa_conf = get_model_predictions(
                        sgd_model, pa_model, vectorizer, news_text
                    )
                    output_widget.insert(tk.END, "\n📊 Final Retrained Model Predictions:\n")
                    output_widget.insert(tk.END, f"🔄 SGD Classifier: {sgd_pred} (Confidence: {sgd_conf*100:.2f}%)\n")
                    output_widget.insert(tk.END, f"🚀 Passive Aggressive: {pa_pred} (Confidence: {pa_conf*100:.2f}%)\n")
                    output_widget.see(tk.END)
        else:
            output_widget.insert(tk.END, "\n⚠️ AI disagrees with Passive Aggressive and supports SGD: This might be Real News\n")
            output_widget.insert(tk.END, "No model retraining needed in this case.\n")
            output_widget.see(tk.END)
    else:
        if messagebox.askyesno("Cross-Check", "Would you like to cross-check with AI?"):
            sgd_model, pa_model, vectorizer = handle_ai_cross_check()

    if messagebox.askyesno("Follow-Up Question", "Would you like to ask a question about this news?"):
        question = tk.simpledialog.askstring("Follow-Up Question", "Enter your question:")
        if question:
            output_widget.insert(tk.END, "\n🔍 Getting answer from AI...\n")
            output_widget.see(tk.END)
            answer = analyze_with_ai(news_text, is_followup=True, user_question=question)
            output_widget.insert(tk.END, f"🧠 AI Response:\n{answer}\n")
            output_widget.see(tk.END)

    output_widget.insert(tk.END, "\n📰 Ready for next analysis.\n")
    output_widget.see(tk.END)
    return sgd_model, pa_model, vectorizer

def test_groq_connection():
    print("\n🔍 Testing Groq API connection...")
    test_message = "Hello, this is a test message to check if the Groq API is working."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_message}
    ]
    
    response = call_groq_api(messages, temperature=0.1, max_tokens=50)
    
    if response:
        print("✅ Groq API connection successful!")
        print(f"🔄 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return True
    else:
        print("❌ Failed to connect to Groq API")
        return False

# --- Tkinter GUI ---
class FakeNewsDetectorGUI:
    def __init__(self, root, sgd_model, pa_model, vectorizer):
        self.root = root
        self.root.title("Fake News Detector")
        self.root.geometry("600x700")
        self.sgd_model = sgd_model
        self.pa_model = pa_model
        self.vectorizer = vectorizer
        
        tk.Label(root, text="Enter News Text, URL, or Select Image:").pack(pady=5)
        self.input_entry = tk.Entry(root, width=60)
        self.input_entry.pack(pady=5)
        
        tk.Label(root, text="Or Select Image:").pack(pady=5)
        self.image_entry = tk.Entry(root, width=60)
        self.image_entry.pack(pady=5)
        tk.Button(root, text="Browse Image", command=self.browse_image).pack(pady=5)
        
        tk.Button(root, text="Analyze", command=self.analyze).pack(pady=10)
        tk.Button(root, text="Retrain Models", command=self.retrain_models).pack(pady=5)
        tk.Button(root, text="Clear Output", command=self.clear_output).pack(pady=5)
        
        tk.Label(root, text="Results:").pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
        self.output_text.pack(pady=5)
        
        self.output_text.insert(tk.END, "🔍 Fake News Detection System\n")
        self.output_text.insert(tk.END, "============================\n")
        self.output_text.insert(tk.END, "Enter news text, a URL, or select an image to analyze.\n")
        self.output_text.see(tk.END)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, file_path)

    def analyze(self):
        input_text = self.input_entry.get().strip()
        image_path = self.image_entry.get().strip()
        
        if not (input_text or image_path):
            messagebox.showerror("Input Error", "Please provide news text, a URL, or an image.")
            return
        
        if image_path:
            self.output_text.insert(tk.END, f"\n🖼️ Extracting text from image: {image_path}\n")
            self.output_text.see(tk.END)
            news_text = extract_text_from_image(image_path)
            if news_text:
                self.output_text.insert(tk.END, f"\n📝 Extracted Text: {news_text}\n")
                self.sgd_model, self.pa_model, self.vectorizer = process_news(
                    news_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                )
            else:
                self.output_text.insert(tk.END, "⚠️ Cannot proceed with analysis due to image processing failure.\n")
                self.output_text.see(tk.END)
        elif input_text:
            if is_url(input_text):
                self.output_text.insert(tk.END, f"\n🌐 Fetching content from URL: {input_text}\n")
                self.output_text.see(tk.END)
                news_text = fetch_url_content(input_text)
                if news_text:
                    self.sgd_model, self.pa_model, self.vectorizer = process_news(
                        news_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                    )
                else:
                    self.output_text.insert(tk.END, "⚠️ Cannot proceed with analysis due to URL fetch failure.\n")
                    self.output_text.see(tk.END)
            else:
                self.output_text.insert(tk.END, f"\n📝 Processing news text: {input_text}\n")
                self.output_text.see(tk.END)
                self.sgd_model, self.pa_model, self.vectorizer = process_news(
                    input_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                )

    def retrain_models(self):
        if messagebox.askyesno("Retrain Models", "Retrain models from scratch? This may take a while."):
            self.output_text.insert(tk.END, "\n🧠 Training new models from scratch...\n")
            self.output_text.see(tk.END)
            self.sgd_model, self.pa_model, self.vectorizer = train_models()
            self.output_text.insert(tk.END, "✅ Models retrained successfully.\n")
            self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "🔍 Fake News Detection System\n")
        self.output_text.insert(tk.END, "============================\n")
        self.output_text.insert(tk.END, "Enter news text, a URL, or select an image to analyze.\n")
        self.output_text.see(tk.END)

# --- Flask API Endpoint ---
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        class StringOutput:
            def __init__(self):
                self.output = []
            def insert(self, _, text):
                self.output.append(text)
            def see(self, _):
                pass
        output_widget = StringOutput()

        sgd_model, pa_model, vectorizer = load_models()

        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                news_text = extract_text_from_image(image_path)
                if not news_text:
                    return jsonify({'error': 'Failed to extract text from image'}), 400
            else:
                return jsonify({'error': 'Invalid image file'}), 400
        elif 'url' in request.form and request.form['url']:
            url = request.form['url']
            if is_url(url):
                news_text = fetch_url_content(url)
                if not news_text:
                    return jsonify({'error': 'Failed to fetch content from URL'}), 400
            else:
                return jsonify({'error': 'Invalid URL'}), 400
        elif 'text' in request.form and request.form['text']:
            news_text = request.form['text']
        else:
            return jsonify({'error': 'No valid input provided'}), 400

        sgd_model, pa_model, vectorizer = process_news(news_text, sgd_model, pa_model, vectorizer, output_widget)

        return jsonify({
            'results': output_widget.output,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument("--train", action="store_true", help="Train new models")
    parser.add_argument("--force-retrain", action="store_true", help="Force training new models")
    parser.add_argument("--input", type=str, help="News text or URL to analyze")
    parser.add_argument("--image", type=str, help="Path to image containing news text")
    parser.add_argument("--no-db", action="store_true", help="Skip database connection attempt")
    parser.add_argument("--test-groq", action="store_true", help="Test Groq API connection")
    parser.add_argument("--api", action="store_true", help="Run as API server")
    args = parser.parse_args()

    print("\n🔍 Fake News Detection System")
    print("============================")

    if not args.no_db:
        setup_database()

    if args.test_groq:
        test_groq_connection()

    if args.api:
        print("\n🚀 Starting Flask API server...")
        app.run(host='0.0.0.0', port=4000, debug=True)
    else:
        if args.train or args.force_retrain:
            print("\n🧠 Training new models from scratch...")
            sgd_model, pa_model, vectorizer = train_models()
        else:
            print("\n🔄 Loading existing models...")
            sgd_model, pa_model, vectorizer = load_models()

        if args.input or args.image:
            output_widget = scrolledtext.ScrolledText()
            current_models = (sgd_model, pa_model, vectorizer)
            if args.input:
                if is_url(args.input):
                    print(f"\n🌐 Fetching content from URL: {args.input}")
                    news_text = fetch_url_content(args.input)
                    if news_text:
                        sgd_model, pa_model, vectorizer = process_news(news_text, *current_models, output_widget)
                    else:
                        print("⚠️ Cannot proceed with analysis due to URL fetch failure.")
                else:
                    print(f"\n📝 Processing news text: {args.input}")
                    sgd_model, pa_model, vectorizer = process_news(args.input, *current_models, output_widget)
            elif args.image:
                print(f"\n🖼️ Extracting text from image: {args.image}")
                news_text = extract_text_from_image(args.image)
                if news_text:
                    print(f"\n📝 Extracted Text: {news_text}")
                    sgd_model, pa_model, vectorizer = process_news(news_text, *current_models, output_widget)
                else:
                    print("⚠️ Cannot proceed with analysis due to image processing failure.")
        else:
            root = tk.Tk()
            app = FakeNewsDetectorGUI(root, sgd_model, pa_model, vectorizer)
            root.mainloop()

if __name__ == "__main__":
    groq_available = test_groq_connection()
    if not groq_available:
        print("\n⚠️ Groq API is not available or key is invalid.")
        print("The system will use rule'system will use rule-based analysis for AI features.")
    
    main()