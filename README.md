# 🏠 Estate Genie — AI Property Search (UK Rentals)

Natural language search over **147,000+ UK rental listings**, powered by FAISS + SentenceTransformers and optionally GPT-4o-mini.

---

## 🚀 Deploy to Streamlit Cloud

### 1. Push to GitHub
```bash
git init && git add . && git commit -m "Estate Genie"
git remote add origin https://github.com/YOUR_USERNAME/estate-genie.git
git push -u origin main
```

### 2. Go to share.streamlit.io
- New app → connect repo → set main file to `app.py` → Deploy

### 3. Add OpenAI key (optional — for AI summaries)
Settings → Secrets:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
Without a key the app still works with template summaries.

---

## 🖥 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
```
estate_genie/
├── app.py               # Streamlit UI
├── rag_engine.py        # RAG pipeline + column mapping
├── requirements.txt
├── data/
│   └── properties.csv   # 147k UK rental listings
└── .streamlit/
    └── secrets.toml
```

---

## 💬 Sample Queries
- `2-bedroom flat under £1,500/mo`
- `Cheapest studios in Westminster`
- `New build apartments in Birmingham`
- `Average rent for 3-bedroom houses`
- `Compare flat vs house rents`
- `Which area has the most crime?`
- `Top 5 most expensive listings`
- `Low flood risk 2-bed under £1,200`

---

## 📊 Dataset Columns
| Column | Description |
|---|---|
| type | Property type (Flat, Terraced, Detached…) |
| bedrooms / bathrooms | Room counts |
| price | Monthly rent in GBP (£) |
| listing_update_date | Date of listing |
| property_type_full_description | Full text description |
| flood_risk | Low / Medium / High / Unknown |
| is_new_home | New build flag |
| laua | ONS local authority code |
| crime_score_weight | Crime score 1–10 |
| address | Area/city name (used as neighbourhood) |
