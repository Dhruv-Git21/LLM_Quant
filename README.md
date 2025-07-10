# Residual Return Forecasting and Market-Neutral Portfolio Construction via LLM-Augmented Machine Learning

This project implements a research-grade, end-to-end quant pipeline for constructing a long-short, market-neutral equity portfolio. 
The workflow fuses LLM-derived sentiment (e.g., FinBERT), technical indicators, and fundamental features, 
using machine learning to predict residual returns and generate robust alpha.

## 📂 Structure
llm_residual_arbitrage/
├── data/
│ ├── raw/ # Raw (downloaded) data
│ └── processed/ # Cleaned/features
├── notebooks/ # Stepwise Jupyter notebooks for each phase
├── src/ # All reusable code modules
├── models/ # Trained model files (joblib/pkl)
├── requirements.txt
└── research_report.pdf


## 🚦 Workflow

1. **Data Collection:**  `notebooks/01_data_collection.ipynb` — Pull OHLCV, Fama-French, news.
2. **Feature Engineering:**  `notebooks/02_feature_engineering.ipynb` — Technical & fundamental features.
3. **Sentiment Extraction:**  `notebooks/03_sentiment_extraction.ipynb` — LLM/NLP pipeline.
4. **Model Training:**  `notebooks/04_model_training.ipynb` — Residual regression & ML.
5. **Backtesting:**  `notebooks/05_backtesting.ipynb` — Portfolio simulation.
6. **Reporting:**  `notebooks/06_reporting.ipynb` — Visuals, tables, export to PDF.

## 🛠️ Key Features

- Automated, reproducible pipeline
- Modular codebase for research or production
- Machine learning with technical, fundamental, and NLP features
- LLM-powered sentiment for alpha generation
- Portfolio backtesting and performance analytics

## 🚀 Getting Started

1. Install dependencies:  
   `pip install -r requirements.txt`
2. Run notebooks in order, starting with data collection.
3. Add more tickers/news as desired for deeper backtests.

---

## 📞 Contact / Credits

Project by Dhruv Goyal.  
Inspired by modern ML and NLP applications in quantitative finance.
