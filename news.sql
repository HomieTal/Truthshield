CREATE DATABASE IF NOT EXISTS fake_news_project;
USE fake_news_project;

CREATE TABLE IF NOT EXISTS news_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255),
    title TEXT,
    description TEXT,
    content TEXT,
    published_at DATETIME,
    fetched_from VARCHAR(50) -- shows from which API (e.g., NewsAPI, GNews)
);
