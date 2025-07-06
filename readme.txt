# Fake News Classifier (NLP + Streamlit)

This project detects whether a given news article is real or fake using a Logistic Regression model trained on TF-IDF vectorized text. It includes a live Streamlit app for interactive use.

## Project Structure
- `data/True.csv`, `Fake.csv` → Raw datasets
- `notebooks/fake_news_model.ipynb` → Training code
- `model/` → Saved model & vectorizer
- `app/app.py` → Streamlit UI

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
