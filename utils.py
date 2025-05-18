import os
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
import requests
from bs4 import BeautifulSoup
import easyocr
import cv2
from functools import lru_cache
import joblib
from itertools import cycle

# Define Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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
UPLOAD_FOLDER = "Uploads"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

# --- API Key Management ---
API_KEYS = []
API_KEY_CYCLE = None

def load_api_keys(file_path='API.txt'):
    """Load API keys from a text file."""
    global API_KEYS, API_KEY_CYCLE
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                API_KEYS = [line.strip() for line in f if line.strip()]
            if not API_KEYS:
                print("⚠️ No valid API keys found in api.txt. Falling back to rule-based analysis.")
                return False
            API_KEY_CYCLE = cycle(API_KEYS)  # Create an iterator to cycle through keys
            print(f"✅ Loaded {len(API_KEYS)} API keys from {file_path}")
            return True
        else:
            print("⚠️ api.txt file not found. Falling back to rule-based analysis.")
            return False
    except Exception as e:
        print(f"❌ Error loading API keys: {str(e)}. Falling back to rule-based analysis.")
        return False

def get_next_api_key():
    """Get the next API key from the cycle."""
    global API_KEY_CYCLE
    if API_KEY_CYCLE is None:
        load_api_keys()
    return next(API_KEY_CYCLE) if API_KEY_CYCLE else None

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

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
    fake_news = []

    # Load data from True and Fake folders
    true_folder = "True"
    fake_folder = "Fake"
    
    # Process True folder
    if os.path.exists(true_folder):
        print(f"ℹ️ Loading real news from folder: {true_folder}")
        for filename in os.listdir(true_folder):
            file_path = os.path.join(true_folder, filename)
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            real_news.append(text)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for col in ['title', 'text']:
                        if col in df.columns:
                            texts = df[col].dropna().tolist()
                            real_news.extend([str(t) for t in texts if str(t).strip()])
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        print(f"✅ Read {len(real_news)} real news articles from {true_folder}")
    else:
        print(f"⚠️ True folder not found at {true_folder}")

    # Process Fake folder
    if os.path.exists(fake_folder):
        print(f"ℹ️ Loading fake news from folder: {fake_folder}")
        for filename in os.listdir(fake_folder):
            file_path = os.path.join(fake_folder, filename)
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            fake_news.append(text)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for col in ['title', 'text']:
                        if col in df.columns:
                            texts = df[col].dropna().tolist()
                            fake_news.extend([str(t) for t in texts if str(t).strip()])
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        print(f"✅ Read {len(fake_news)} fake news articles from {fake_folder}")
    else:
        print(f"⚠️ Fake folder not found at {fake_folder}")

    # Fallback to original CSV files if folders are empty or not found
    if not real_news:
        print("ℹ️ No real news from True folder. Trying True.csv...")
        try:
            true_df = pd.read_csv("True.csv")
            if not true_df.empty:
                real_news = true_df['title'].dropna().tolist()
                print(f"✅ Read {len(real_news)} real news articles from True.csv")
        except Exception as e:
            print(f"❌ Error loading True.csv: {e}")
            print("⚠️ No real news data found. Creating dummy data for demo purposes.")
            real_news = [
                "NASA Confirms Evidence of Water on Mars",
                "Scientists Develop New Cancer Treatment",
                "Federal Reserve Announces Interest Rate Decision",
                "New Study Links Exercise to Longevity",
                "Stock Market Closes at Record High"
            ]
            print(f"✅ Read {len(real_news)} real news articles from dummy data")

    if not fake_news:
        print("ℹ️ No fake news from Fake folder. Trying Fake.csv...")
        try:
            fake_df = pd.read_csv("Fake.csv")
            fake_news = fake_df['title'].dropna().tolist()
            print(f"✅ Read {len(fake_news)} fake news articles from Fake.csv")
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
            print(f"✅ Read {len(fake_news)} fake news articles from dummy data")

    # Load database if available
    if cursor:
        try:
            cursor.execute("SELECT title, description, content FROM news_articles")
            records = cursor.fetchall()
            db_real_news = [' '.join(filter(None, r)) for r in records if any(r)]
            real_news.extend(db_real_news)
            print(f"✅ Read {len(db_real_news)} real news articles from database")
        except Exception as e:
            print(f"❌ Error loading from database: {e}")

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
    
    print(f"ℹ️ Prepared {len(real_news)} real and {len(fake_news)} fake news articles for training")

    texts = real_news + fake_news
    labels = [1]*len(real_news) + [0]*len(fake_news)
    texts = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"ℹ️ Training set: {X_train.shape[0]} articles ({len([l for l in y_train if l == 1])} real, {len([l for l in y_train if l == 0])} fake)")
    print(f"ℹ️ Test set: {X_test.shape[0]} articles ({len([l for l in y_test if l == 1])} real, {len([l for l in y_test if l == 0])} fake)")

    print("\n🔄 Training SGD Classifier model...")
    print(f"🔄 SGD Classifier: Training on {X_train.shape[0]} articles")
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
    print(f"🚀 Passive Aggressive Classifier: Training on {X_train.shape[0]} articles")
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

# --- Save Models ---
def save_models(sgd_model, pa_model, vectorizer):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(sgd_model, os.path.join(MODELS_DIR, 'sgd_model.pkl'))
    joblib.dump(pa_model, os.path.join(MODELS_DIR, 'pa_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    print(f"✅ Models saved to {MODELS_DIR}")

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
    if not API_KEYS:
        if not load_api_keys():
            return None
    
    for _ in range(len(API_KEYS)):
        api_key = get_next_api_key()
        if not api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
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
                print(f"✅ Successful API call with key ending in {api_key[-4:]}")
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"⚠️ API response is empty or invalid with key ending in {api_key[-4:]}")
                continue
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request Error with key ending in {api_key[-4:]}: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Status code: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            continue
    print("❌ All API keys failed. Falling back to rule-based analysis.")
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

def process_news(news_text, sgd_model, pa_model, vectorizer, output_widget, include_ai=False):
    output_widget.insert_model(None, f"📰 Analyzing: {news_text}\n")
    
    start_time = time.time()
    sgd_pred, sgd_conf, pa_pred, pa_conf = get_model_predictions(
        sgd_model, pa_model, vectorizer, news_text
    )
    prediction_time = time.time() - start_time
    
    output_widget.insert_model(None, f"🔄 SGD Classifier: {sgd_pred} (Confidence: {sgd_conf*100:.2f}%)\n")
    output_widget.insert_model(None, f"🚀 Passive Aggressive: {pa_pred} (Confidence: {pa_conf*100:.2f}%)\n")
    output_widget.insert_model(None, f"⏱️ Prediction completed in {prediction_time:.4f} seconds\n")
    
    if include_ai:
        output_widget.insert_ai(None, "🧠 Cross-checking with AI...\n")
        ai_check = analyze_with_ai(news_text)
        output_widget.insert_ai(None, f"🔄 AI Analysis:\n{ai_check}\n")
        
        agreement, ai_verdict = check_ai_agreement(ai_check, sgd_pred, pa_pred)
        if agreement:
            output_widget.insert_ai(None, f"✅ AI ({ai_verdict}) agrees with model prediction.\n")
        else:
            output_widget.insert_ai(None, f"❌ AI ({ai_verdict}) disagrees with model predictions.\n")

    return sgd_model, pa_model, vectorizer

def test_groq_connection():
    print("\n🔍 Testing Groq API connection...")
    if not API_KEYS:
        if not load_api_keys():
            return False
    
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