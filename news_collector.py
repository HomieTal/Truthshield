"""
News Collector - Fetches news articles from various APIs and stores them in a database
"""
import requests
import mysql.connector
import re
import feedparser
from datetime import datetime

# --- DB Connection ---
db = mysql.connector.connect(host="localhost", user="root", password="qwerty00", database="fake_news_project")
cursor = db.cursor()

# --- API Keys ---
API_KEYS = {
    'newsapi': 'df605d8040b54e0c9de27cbc2731007f',
    'gnews': 'a200328d8113c4c48d088ff20359ba41',
    'mediastack': '3dddc2348489a1b330b363d509b8059a',
    'newsdata': 'pub_83766cc1637b7904db1f4f427e162ae5ded42',
    'currents': 'rbkW2YNmyUb7y1BGTO2utrQQVyg4dL35aFBlyP6lc-SFngk9',
    'nytimes': 'utykOKg1oQgEwIu4hgB32SYLYNdDyGQW',
}

# --- Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def insert_article(source_name, title, description, content, published_at, fetched_from):
    try:
        if '+' in published_at:
            published_at = published_at.split('+')[0].strip()
        cursor.execute(
            "INSERT INTO news_articles (source_name, title, description, content, published_at, fetched_from) VALUES (%s, %s, %s, %s, %s, %s)",
            (source_name, title, description, content, published_at, fetched_from)
        )
        db.commit()
    except mysql.connector.Error as err:
        print(f"❌ DB Insert Error: {err}")

# --- Unified Fetch Function ---
def fetch_from_api(api_name):
    try:
        if api_name == 'newsapi':
            url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEYS['newsapi']}&pageSize=50"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('articles', []):
                    insert_article(
                        article['source']['name'], 
                        article.get('title', ''), 
                        article.get('description', ''),
                        article.get('content', ''), 
                        article.get('publishedAt', '').replace('T', ' ').replace('Z', ''), 
                        'NewsAPI'
                    )
        elif api_name == 'gnews':
            url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=50&token={API_KEYS['gnews']}"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('articles', []):
                    insert_article(
                        article['source']['name'], 
                        article.get('title', ''), 
                        article.get('description', ''),
                        article.get('content', ''), 
                        article.get('publishedAt', '').replace('T', ' ').replace('Z', ''), 
                        'GNews'
                    )
        elif api_name == 'mediastack':
            url = f"http://api.mediastack.com/v1/news?access_key={API_KEYS['mediastack']}&countries=in&languages=en&limit=50"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('data', []):
                    insert_article(
                        article.get('source', ''), 
                        article.get('title', ''), 
                        article.get('description', ''),
                        article.get('description', ''), 
                        article.get('published_at', ''), 
                        'MediaStack'
                    )
        elif api_name == 'newsdata':
            url = f"https://newsdata.io/api/1/news?apikey={API_KEYS['newsdata']}&country=in&language=en&category=top"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('results', []):
                    insert_article(
                        article.get('source_id', ''), 
                        article.get('title', ''), 
                        article.get('description', ''),
                        article.get('content', ''), 
                        article.get('pubDate', ''), 
                        'NewsData.io'
                    )
        elif api_name == 'google_rss':
            feed = feedparser.parse("https://news.google.com/rss/search?q=india")
            for entry in feed.entries:
                published = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d %H:%M:%S')
                insert_article(
                    "Google News", 
                    entry.title, 
                    entry.get('summary', ''), 
                    entry.get('summary', ''), 
                    published, 
                    'GoogleRSS'
                )
        elif api_name == 'currents':
            url = f"https://api.currentsapi.services/v1/latest-news?apiKey={API_KEYS['currents']}&language=en&country=IN"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('news', []):
                    published = article.get('published', '').split('+')[0].strip()
                    insert_article(
                        article.get('author', ''), 
                        article.get('title', ''), 
                        article.get('description', ''),
                        article.get('description', ''), 
                        published, 
                        'CurrentsAPI'
                    )
        elif api_name == 'nytimes':
            url = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={API_KEYS['nytimes']}"
            response = requests.get(url)
            if response.status_code == 200:
                for article in response.json().get('results', []):
                    insert_article(
                        "NYTimes", 
                        article.get('title', ''), 
                        article.get('abstract', ''),
                        article.get('abstract', ''), 
                        article.get('published_date', ''), 
                        'NYTimes'
                    )
        print(f"✅ {api_name.capitalize()} articles fetched.")
    except Exception as e:
        print(f"❌ {api_name.capitalize()}: {str(e)}")

def fetch_all_news():
    """Fetch news from all configured APIs"""
    apis = ['newsapi', 'gnews', 'mediastack', 'newsdata', 'google_rss', 'currents', 'nytimes']
    for api in apis:
        fetch_from_api(api)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="News Article Collector")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh news articles")
    parser.add_argument("--api", type=str, help="Fetch from specific API")
    args = parser.parse_args()
    
    if args.fetch:
        if args.api:
            print(f"🔄 Fetching news from {args.api}...")
            fetch_from_api(args.api)
        else:
            print("🔄 Fetching news from all sources...")
            fetch_all_news()
    else:
        print("No action specified. Use --fetch to collect news.")

if __name__ == "__main__":
    main()
