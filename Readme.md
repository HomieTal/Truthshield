📢 Fake News Detection System



🌟 Overview
Welcome to the Fake News Detection System, a powerful tool designed to combat misinformation by analyzing news articles for authenticity. Whether you're inputting text, URLs, or images, this system leverages machine learning and optional AI analysis to deliver accurate results. It features a user-friendly Tkinter GUI and a robust Flask-based web API, making it accessible for both casual users and developers.

✨ Key Features

Flexible Input Methods:
📝 Direct text input
🌐 URLs (scrapes content automatically)
🖼️ Images (extracts text via OCR)


Advanced Machine Learning:
🚀 SGD Classifier
⚡ Passive Aggressive Classifier
📊 TF-IDF Vectorizer for text processing


AI-Powered Analysis:
🧠 Optional Groq API integration for deeper insights


Data Sources:
📂 Local True and Fake folders (text/CSV)
🗄️ MySQL database for news articles
🌍 Multiple news APIs for real-time data


Interactive Interfaces:
🖥️ Tkinter GUI for intuitive use
🌐 Flask API for programmatic access


Model Management:
🔄 Retrain models with fresh data
💾 Save and load trained models




🛠️ Requirements

Python: 3.8 or higher
MySQL: Optional, for database storage
Groq API Key: Optional, for AI-powered analysis
Dependencies: Listed in requirements.txt


🚀 Installation
1. Clone the Repository
git clone <repository-url>
cd fake-news-detection

2. Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

Or use the provided setup.bat for Windows:
setup.bat

4. Configure MySQL Database (Optional)

Install MySQL and create a database:CREATE DATABASE IF NOT EXISTS fake_news_project;


Update credentials in utils.py and news_collector.py (default: user=root, password=qwerty00).
Create the news_articles table:USE fake_news_project;
CREATE TABLE IF NOT EXISTS news_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255),
    title TEXT,
    description TEXT,
    content TEXT,
    published_at DATETIME,
    fetched_from VARCHAR(50)
);



5. Set Up Groq API Key (Optional)

Create API.txt in the project root.
Add your Groq API key(s), one per line.
Without a key, the system uses rule-based analysis.

6. Prepare Data

Place real and fake news in True and Fake folders (text or CSV files).
Alternatively, use True.csv and Fake.csv or fetch articles via news_collector.py.


🎮 Usage
Launch the GUI
Run the Tkinter-based interface:
python app.py


Features:
Enter text, a URL, or select an image.
Click Analyze for model predictions.
Use Retrain Models to update models.
Click Clear Output to reset the display.



Start the Flask API
Run the web server:
python app.py --api


Server: Runs at http://0.0.0.0:4000
Endpoints:
GET /: Serves index.html (if available)
POST /api/analyze:
Input: text, url, or image (multipart form-data)
Optional: include_ai (boolean) for Groq analysis
Output: JSON with model and AI results


POST /api/retrain: Retrains models


Example Request:curl -X POST -F "text=Sample news text" -F "include_ai=true" http://localhost:4000/api/analyze



Command-Line Options
Analyze specific inputs or manage the system:
python app.py [options]


--train: Train new models
--force-retrain: Force model retraining
--input <text-or-url>: Analyze text or URL
--image <path>: Analyze image text
--no-db: Skip database connection
--test-groq: Test Groq API
--api: Run Flask server

Fetch News Articles
Collect news from APIs:
python news_collector.py --fetch


Use --api <api_name> for specific APIs (e.g., newsapi, gnews, mediastack, newsdata, google_rss, currents, nytimes).
Without --api, fetches from all APIs.


📂 Project Structure
fake-news-detection/
├── app.py                # Main app (GUI + API)
├── news_collector.py     # Fetches news articles
├── utils.py              # Core utilities
├── setup.bat             # Windows setup script
├── requirements.txt      # Dependencies
├── API.txt               # Groq API keys
├── True/                 # Real news data
├── Fake/                 # Fake news data
├── True.csv              # Sample real news
├── Fake.csv              # Sample fake news
├── models/               # Trained models
├── Uploads/              # Image uploads
├── static/               # Flask static files
└── README.md             # You're here!


📚 Data Sources

Local: True and Fake folders (text/CSV)
Database: news_articles table in MySQL
APIs:
NewsAPI
GNews
MediaStack
NewsData.io
Google News RSS
CurrentsAPI
NYTimes




🛠️ Notes

Groq API: Requires API.txt with valid keys; otherwise, uses rule-based fallback.
Database: If MySQL is unavailable, relies on local data.
OCR: Powered by EasyOCR, optimized for speed.
Models: Trained on local data, database, or Groq feedback (if enabled).
API Keys: Replace news API keys in news_collector.py with your own.


⚠️ Troubleshooting

Dependencies: Run pip install -r requirements.txt or setup.bat.
Database: Verify MySQL credentials; use --no-db if issues persist.
Groq API: Check API.txt and test with --test-groq.
Models: Retrain with --train if loading fails.
Images: Use PNG, JPG, JPEG, or GIF with clear text.


🌈 Future Enhancements

📰 Expand news API support
🤖 Integrate multi-model AI ensemble
🖼️ Improve OCR for low-quality images
⏰ Add real-time news monitoring
📢 Incorporate user feedback for training


📜 License
Licensed under the MIT License.

📬 Contact
Have questions or ideas? Open an issue or submit a pull request on the repository!

Stay Informed, Stay Truthful! 🚨
