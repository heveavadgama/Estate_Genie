"""
Estate Genie — AI Property Search (UK Rental Dataset)
"""

import os
import re
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Estate Genie",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.eg-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.eg-hero::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(229,160,55,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.eg-hero h1 {
    font-family: 'Playfair Display', serif; font-size: 2.4rem;
    font-weight: 700; color: #e5a037; margin: 0 0 0.3rem; line-height: 1.1;
}
.eg-hero p { color: rgba(255,255,255,0.72); font-size: 1rem; margin: 0; font-weight: 300; }

.prop-card {
    background: #ffffff; border: 1px solid #e8e8e8; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: box-shadow 0.2s;
}
.prop-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
.prop-card-header { display: flex; justify-content: space-between; align-items: flex-start; }
.prop-price {
    font-family: 'Playfair Display', serif; font-size: 1.5rem;
    font-weight: 700; color: #e5a037;
}
.prop-pcm { font-size: 0.78rem; color: #999; font-weight: 400; }
.prop-area { font-weight: 500; font-size: 1rem; color: #1a1a2e; }
.prop-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; background: #f0f4ff;
    color: #3a5bd9; border: 1px solid #d0daff; margin-right: 4px;
}
.prop-badge-gold {
    background: #fff8e7; color: #b07800; border: 1px solid #fde68a;
}
.prop-stats { display: flex; gap: 1rem; margin: 0.8rem 0 0.5rem; flex-wrap: wrap; }
.prop-stat { color: #444; font-size: 0.9rem; }
.prop-desc { color: #666; font-size: 0.88rem; line-height: 1.5; margin-top: 0.6rem; }
.ai-summary {
    background: linear-gradient(135deg, #fffbf0, #fff8e7);
    border-left: 4px solid #e5a037; border-radius: 8px;
    padding: 1.1rem 1.3rem; margin-bottom: 1.4rem;
    font-size: 0.95rem; color: #333; line-height: 1.6;
}
.crime-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 5px;
}
.stTextInput > div > div > input {
    border-radius: 10px !important; border: 2px solid #e8e8e8 !important;
    font-size: 1rem !important; padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e5a037 !important;
    box-shadow: 0 0 0 3px rgba(229,160,55,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading — file uploader + optional bundled CSV ──────────────────────
import pathlib, io, hashlib

@st.cache_resource(show_spinner="🏠 Indexing listings…")
def load_engine_from_bytes(csv_bytes: bytes, _cache_key: str):
    """Load engine from raw CSV bytes (uploaded or read from disk)."""
    import tempfile, os
    from rag_engine import EstateGenieRAG
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = tmp.name
    try:
        engine = EstateGenieRAG(tmp_path)
    finally:
        os.unlink(tmp_path)
    return engine

def _try_bundled_csv() -> bytes | None:
    """Return bytes if a valid bundled CSV exists, else None."""
    candidates = [
        pathlib.Path(__file__).parent / "data" / "properties.csv",
        pathlib.Path("data") / "properties.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            return p.read_bytes()
    return None

def get_engine_and_stats():
    bundled = _try_bundled_csv()

    if bundled:
        cache_key = hashlib.md5(bundled[:4096]).hexdigest()
        engine = load_engine_from_bytes(bundled, cache_key)
        return engine, engine.get_stats(), None   # None = no uploader needed

    # No valid bundled file — show uploader
    st.markdown("""
<div class="eg-hero">
  <h1>🏠 Estate Genie</h1>
  <p>Search UK rental listings in plain English — powered by AI.</p>
</div>
""", unsafe_allow_html=True)

    st.warning(
        "**Dataset not found in the repo.** Upload your `properties.csv` to get started. "
        "The file is cached for your session — no need to re-upload on every search."
    )
    uploaded = st.file_uploader(
        "Upload properties.csv",
        type=["csv"],
        help="Your property dataset with columns: type, bedrooms, bathrooms, price, "
             "listing_update_date, property_type_full_description, flood_risk, "
             "is_new_home, laua, crime_score_weight, address",
    )
    if uploaded is None:
        st.info("👆 Upload your CSV file to begin.")
        st.stop()

    csv_bytes = uploaded.read()
    if len(csv_bytes) < 500:
        st.error("The uploaded file looks empty or too small. Please upload the full dataset.")
        st.stop()

    with st.spinner("🏠 Indexing your listings — this takes ~30 seconds on first load…"):
        cache_key = hashlib.md5(csv_bytes[:4096]).hexdigest()
        try:
            engine = load_engine_from_bytes(csv_bytes, cache_key)
        except Exception as e:
            st.error(f"**Failed to load dataset:** {e}")
            st.stop()

    return engine, engine.get_stats(), uploaded.name

@st.cache_resource(show_spinner="🤖 Connecting to AI…")
def get_openai_client():
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None

engine, stats, _uploaded_name = get_engine_and_stats()
openai_client = get_openai_client()

# ── AI summary ────────────────────────────────────────────────────────────────
def generate_ai_summary(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return "No matching properties found."
    if openai_client:
        try:
            sample = results.head(5)[
                ["neighbourhood", "price", "bedrooms", "bathrooms", "property_type"]
            ].to_dict("records")
            prompt = (
                f'User searched: "{query}"\n'
                f"Top results (first 5 of {len(results)}):\n{sample}\n\n"
                "In 2-3 sentences summarise what was found and highlight the best value. "
                "Note prices are monthly rent in GBP. Be concise."
            )
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    n = len(results)
    avg = results["price"].mean()
    best = results.iloc[0]
    return (
        f"Found **{n} listings** matching your search. "
        f"Average rent: **£{avg:,.0f}/mo**. "
        f"Top match: **{best['neighbourhood']}** — "
        f"{best['bedrooms']}-bed {best['property_type']} at **£{best['price']:,}/mo**."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    st.caption("NL query overrides these when detected.")

    sb_type = st.selectbox("Property Type", ["Any"] + stats["property_types"])
    sb_beds = st.selectbox("Min Bedrooms", ["Any", 0, 1, 2, 3, 4, 5, 6])
    sb_baths = st.selectbox("Min Bathrooms", ["Any", 1, 2, 3, 4])

    price_cap = min(stats["max_price"], 20000)
    sb_min_price, sb_max_price = st.slider(
        "Monthly Rent (£)",
        min_value=stats["min_price"],
        max_value=price_cap,
        value=(stats["min_price"], price_cap),
        step=50,
        format="£%d",
    )

    sb_area = st.selectbox(
        "Area", ["Any"] + stats["neighbourhoods"],
        help="Matches the 'address' field (city/borough)"
    )
    sb_new_home = st.checkbox(f"New builds only ({stats['new_home_count']:,} listings)")
    sb_flood = st.selectbox("Flood Risk", ["Any", "Low", "Medium", "High", "Unknown"])
    sb_top_k = st.slider("Max Results", 5, 30, 10)

    st.divider()
    st.markdown("### 💡 Sample Queries")
    samples = [
        "2-bedroom flats under £1,500/mo",
        "Cheapest studios in Westminster",
        "4-bed detached houses in Birmingham",
        "New build apartments under £2,000",
        "Average rent for 3-bedroom houses",
        "Which area has the most crime?",
        "Compare flat vs house rents",
        "Top 5 most expensive listings",
        "Low flood risk 2-bed under £1,200",
    ]
    for sq in samples:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state["prefill_query"] = sq

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eg-hero">
  <h1>🏠 Estate Genie</h1>
  <p>Search 147,000+ UK rental listings in plain English — powered by AI.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Listings", f"{stats['total']:,}")
c2.metric("Avg Rent", f"£{stats['avg_price']:,.0f}/mo")
c3.metric("Median Rent", f"£{stats['median_price']:,.0f}/mo")
c4.metric("Areas", len(stats["neighbourhoods"]))

st.divider()

prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "search",
    value=prefill,
    placeholder='e.g. "2-bed flat in Manchester under £1,200/mo with low crime"',
    label_visibility="collapsed",
)

if not query:
    st.info("👆 Type a question or pick a sample from the sidebar to get started.")
    st.stop()

# Build sidebar filter dict
sidebar_filters: dict = {}
if sb_type != "Any":
    sidebar_filters["property_type"] = sb_type
if sb_beds != "Any":
    sidebar_filters["min_bedrooms"] = int(sb_beds)
if sb_baths != "Any":
    sidebar_filters["bathrooms"] = int(sb_baths)
sidebar_filters["min_price"] = sb_min_price
sidebar_filters["max_price"] = sb_max_price
if sb_area != "Any":
    sidebar_filters["neighbourhood"] = sb_area
if sb_new_home:
    sidebar_filters["new_home"] = True
if sb_flood != "Any":
    sidebar_filters["flood_risk"] = sb_flood

# Search
with st.spinner("🔍 Searching across 147k listings…"):
    results, parsed, analytical_answer = engine.search(query, sidebar_filters, top_k=sb_top_k)

# Query understanding expander
with st.expander("🧠 Query Understanding", expanded=False):
    cols = st.columns(6)
    fields = [
        ("Bedrooms", parsed.get("bedrooms") or parsed.get("min_bedrooms")),
        ("Bathrooms", parsed.get("bathrooms")),
        ("Max Rent", f"£{parsed['max_price']:,}" if parsed.get("max_price") else None),
        ("Min Rent", f"£{parsed['min_price']:,}" if parsed.get("min_price") else None),
        ("Type", parsed.get("property_type")),
        ("New Build", "Yes" if parsed.get("new_home") else None),
    ]
    for col, (lbl, val) in zip(cols, fields):
        col.metric(lbl, str(val) if val is not None else "—")

# Analytical branch
if analytical_answer:
    st.markdown("### 📊 Market Analysis")
    st.markdown(f'<div class="ai-summary">{analytical_answer}</div>', unsafe_allow_html=True)
    st.stop()

# Summary + results
if not results.empty:
    summary = generate_ai_summary(query, results)
    st.markdown(f'<div class="ai-summary">🤖 <strong>AI Summary</strong><br>{summary}</div>',
                unsafe_allow_html=True)

    st.markdown(f"#### Found {len(results)} Listings")

    view_mode = st.radio("View as:", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")

    if view_mode == "Table":
        disp = results[["neighbourhood", "property_type", "price", "bedrooms",
                         "bathrooms", "crime_score", "flood_risk",
                         "is_new_home", "listing_date"]].copy()
        disp["price"] = disp["price"].apply(lambda x: f"£{x:,}/mo")
        disp.columns = ["Area", "Type", "Rent", "Beds", "Baths",
                        "Crime", "Flood Risk", "New Build", "Listed"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        for _, row in results.iterrows():
            crime = float(row.get("crime_score", 5))
            crime_color = "#e74c3c" if crime >= 7 else ("#f39c12" if crime >= 4 else "#27ae60")
            flood = str(row.get("flood_risk", "Unknown"))
            flood_color = {"High": "#e74c3c", "Medium": "#f39c12",
                           "Low": "#27ae60"}.get(flood, "#aaa")
            new_badge = (
                '<span class="prop-badge prop-badge-gold">✨ New Build</span>'
                if row.get("is_new_home") else ""
            )
            desc = str(row.get("description", ""))
            desc_short = desc[:200] + "…" if len(desc) > 200 else desc

            st.markdown(f"""
<div class="prop-card">
  <div class="prop-card-header">
    <div>
      <div class="prop-area">📍 {row['neighbourhood']}</div>
      <div style="margin-top:4px">
        <span class="prop-badge">{row['property_type']}</span>
        {new_badge}
      </div>
    </div>
    <div style="text-align:right">
      <div class="prop-price">£{int(row['price']):,}<span class="prop-pcm">/mo</span></div>
    </div>
  </div>
  <div class="prop-stats">
    <span class="prop-stat">🛏 {int(row['bedrooms'])} bed</span>
    <span class="prop-stat">🚿 {int(row['bathrooms'])} bath</span>
    <span class="prop-stat">
      <span class="crime-dot" style="background:{crime_color}"></span>Crime: {crime:.0f}/10
    </span>
    <span class="prop-stat" style="color:{flood_color}">🌊 Flood: {flood}</span>
    <span class="prop-stat">📅 {row.get('listing_date','')}</span>
  </div>
  <div class="prop-desc">{desc_short}</div>
</div>
""", unsafe_allow_html=True)
else:
    st.warning("🔍 No listings matched your query. Try adjusting filters or rephrasing.")
    st.markdown("**Suggestions:**")
    st.markdown("- Widen your price range")
    st.markdown("- Remove a bedroom or bathroom requirement")
    st.markdown("- Try a different area name")
