"""
Estate Genie — AI Property Search (UK Rental Dataset)
"""

import os
import html
import hashlib
import pathlib
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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hero ── */
.eg-hero {
    background: linear-gradient(135deg, #0f1b2d 0%, #1a2f4e 60%, #0f3460 100%);
    padding: 2.2rem 2.2rem 1.8rem; border-radius: 20px; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.eg-hero::after {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(229,160,55,0.18) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.eg-hero h1 {
    font-family: 'Playfair Display', serif; font-size: 2.6rem;
    font-weight: 700; color: #e5a037; margin: 0 0 0.4rem; line-height: 1.1;
}
.eg-hero p { color: rgba(255,255,255,0.65); font-size: 1rem; margin: 0; }

/* ── Search bar ── */
.stTextInput > div > div > input {
    border-radius: 12px !important; border: 2px solid #e2e8f0 !important;
    font-size: 1.05rem !important; padding: 0.75rem 1.1rem !important;
    background: #fafafa !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e5a037 !important;
    box-shadow: 0 0 0 4px rgba(229,160,55,0.12) !important;
    background: #fff !important;
}

/* ── Property card ── */
.prop-card {
    background: #fff;
    border: 1px solid #eaecf0;
    border-radius: 16px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.18s, transform 0.18s;
}
.prop-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    transform: translateY(-2px);
}
.prop-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.prop-left { flex: 1; min-width: 0; }
.prop-area {
    font-weight: 600; font-size: 1.05rem; color: #1a1a2e;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.prop-price {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700; color: #e5a037;
    white-space: nowrap;
}
.prop-price-sub { font-size: 0.75rem; color: #aaa; font-weight: 400; margin-left: 2px; }
.prop-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.76rem; font-weight: 500; line-height: 1.6;
}
.badge-blue  { background: #eff4ff; color: #3a5bd9; border: 1px solid #c7d7fd; }
.badge-gold  { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.badge-green { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.badge-red   { background: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }
.badge-amber { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.badge-gray  { background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; }
.prop-stats {
    display: flex; flex-wrap: wrap; gap: 0.8rem;
    margin: 0.9rem 0 0.7rem; align-items: center;
}
.prop-stat { color: #555; font-size: 0.88rem; display: flex; align-items: center; gap: 4px; }
.crime-dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; flex-shrink: 0;
}
.prop-divider { border: none; border-top: 1px solid #f1f3f5; margin: 0.7rem 0; }
.prop-desc { color: #6b7280; font-size: 0.86rem; line-height: 1.55; }

/* ── AI Summary box ── */
.ai-box {
    background: linear-gradient(135deg, #fffdf5, #fff8e1);
    border: 1px solid #fde68a; border-left: 4px solid #e5a037;
    border-radius: 12px; padding: 1rem 1.3rem;
    margin-bottom: 1.4rem; font-size: 0.94rem; color: #333; line-height: 1.65;
}
.ai-box strong { color: #b45309; }

/* ── Analytical result box ── */
.analytics-box {
    background: linear-gradient(135deg, #f0f4ff, #e8f0fe);
    border: 1px solid #c7d7fd; border-left: 4px solid #3a5bd9;
    border-radius: 12px; padding: 1.1rem 1.4rem;
    font-size: 0.94rem; color: #1e293b; line-height: 1.8;
}

/* ── Sidebar tweaks ── */
section[data-testid="stSidebar"] { background: #fafbfc; }
.sidebar-header { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94a3b8; margin: 0.8rem 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def e(val) -> str:
    """HTML-escape a value so no raw tags ever leak into the page."""
    return html.escape(str(val))


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🏠 Indexing listings — please wait…")
def load_engine_from_bytes(csv_bytes: bytes, _cache_key: str):
    import tempfile
    from rag_engine import EstateGenieRAG
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = tmp.name
    try:
        eng = EstateGenieRAG(tmp_path)
    finally:
        os.unlink(tmp_path)
    return eng


def _try_bundled_csv():
    for p in [
        pathlib.Path(__file__).parent / "data" / "properties.csv",
        pathlib.Path("data") / "properties.csv",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p.read_bytes()
    return None


def get_engine_and_stats():
    bundled = _try_bundled_csv()
    if bundled:
        key = hashlib.md5(bundled[:4096]).hexdigest()
        eng = load_engine_from_bytes(bundled, key)
        return eng, eng.get_stats()

    # ── Upload screen ─────────────────────────────────────────────────────────
    st.markdown("""
<div class="eg-hero">
  <h1>🏠 Estate Genie</h1>
  <p>AI-powered UK rental search — upload your dataset to begin.</p>
</div>""", unsafe_allow_html=True)

    st.info("**No dataset found in the repo.** Upload your `properties.csv` below — "
            "it's cached for your session so you only do this once.")

    uploaded = st.file_uploader("Upload properties.csv", type=["csv"])
    if uploaded is None:
        st.stop()

    csv_bytes = uploaded.read()
    if len(csv_bytes) < 500:
        st.error("File looks empty. Please upload the full dataset.")
        st.stop()

    with st.spinner("Indexing your listings — ~30 seconds on first load…"):
        key = hashlib.md5(csv_bytes[:4096]).hexdigest()
        try:
            eng = load_engine_from_bytes(csv_bytes, key)
        except Exception as ex:
            st.error(f"Failed to load dataset: {ex}")
            st.stop()

    return eng, eng.get_stats()


@st.cache_resource
def get_openai_client():
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None


engine, stats = get_engine_and_stats()
openai_client = get_openai_client()


# ── AI summary ────────────────────────────────────────────────────────────────
def generate_ai_summary(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return "No matching listings found."
    if openai_client:
        try:
            sample = results.head(5)[
                ["neighbourhood", "price", "bedrooms", "bathrooms", "property_type"]
            ].to_dict("records")
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    f'User searched: "{query}"\n'
                    f"Top {len(sample)} of {len(results)} results:\n{sample}\n\n"
                    "In 2 sentences, summarise what was found and flag the best-value option. "
                    "Prices are monthly rent in GBP £. Be concise."
                )}],
                max_tokens=120, temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    best = results.iloc[0]
    return (
        f"Found **{len(results)} listings** · avg rent **£{results['price'].mean():,.0f}/mo**. "
        f"Top match: **{e(best['neighbourhood'])}** — "
        f"{int(best['bedrooms'])}‑bed {e(best['property_type'])} "
        f"at **£{int(best['price']):,}/mo**."
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Estate Genie")
    st.caption("Natural language property search")
    st.divider()

    st.markdown('<div class="sidebar-header">Property Filters</div>', unsafe_allow_html=True)
    sb_type  = st.selectbox("Type", ["Any"] + stats["property_types"])
    sb_beds  = st.selectbox("Min Bedrooms", ["Any", 0, 1, 2, 3, 4, 5, 6])
    sb_baths = st.selectbox("Min Bathrooms", ["Any", 1, 2, 3, 4])

    price_cap = min(stats["max_price"], 20_000)
    sb_min_p, sb_max_p = st.slider(
        "Monthly Rent (£)", stats["min_price"], price_cap,
        (stats["min_price"], price_cap), step=50, format="£%d",
    )
    sb_area      = st.selectbox("Area", ["Any"] + stats["neighbourhoods"])
    sb_new_home  = st.checkbox(f"New builds only  ({stats['new_home_count']:,})")
    sb_flood     = st.selectbox("Flood Risk", ["Any", "Low", "Medium", "High", "Unknown"])
    sb_top_k     = st.slider("Results to show", 5, 30, 10)

    st.divider()
    st.markdown('<div class="sidebar-header">Sample Queries</div>', unsafe_allow_html=True)
    for sq in [
        "2-bed flat under £1,500/mo",
        "Cheapest studios in Westminster",
        "4-bed detached in Birmingham",
        "New build apartments under £2,000",
        "Average rent for 3-bedroom houses",
        "Which area has the most crime?",
        "Compare flat vs house rents",
        "Top 5 most expensive listings",
        "Low flood risk 2-bed under £1,200",
    ]:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state["prefill_query"] = sq


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="eg-hero">
  <h1>🏠 Estate Genie</h1>
  <p>Search {stats['total']:,} UK rental listings in plain English — powered by AI.</p>
</div>""", unsafe_allow_html=True)

# Stat pills
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Listings",  f"{stats['total']:,}")
c2.metric("Avg Rent",        f"£{stats['avg_price']:,.0f}/mo")
c3.metric("Median Rent",     f"£{stats['median_price']:,.0f}/mo")
c4.metric("Areas Covered",   len(stats["neighbourhoods"]))

st.divider()

# Search box
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "search",
    value=prefill,
    placeholder='🔍  e.g. "2-bed flat in Manchester under £1,200/mo with low crime"',
    label_visibility="collapsed",
)

if not query:
    st.markdown(
        "<p style='color:#94a3b8; text-align:center; padding:2rem 0;'>"
        "Type a question above or choose a sample query from the sidebar to get started.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# Build sidebar filter dict
sb_filters: dict = {}
if sb_type  != "Any": sb_filters["property_type"] = sb_type
if sb_beds  != "Any": sb_filters["min_bedrooms"]  = int(sb_beds)
if sb_baths != "Any": sb_filters["bathrooms"]     = int(sb_baths)
sb_filters["min_price"] = sb_min_p
sb_filters["max_price"] = sb_max_p
if sb_area     != "Any": sb_filters["neighbourhood"] = sb_area
if sb_new_home:          sb_filters["new_home"]       = True
if sb_flood    != "Any": sb_filters["flood_risk"]     = sb_flood

# ── Run search ────────────────────────────────────────────────────────────────
with st.spinner("Searching…"):
    results, parsed, analytical_answer = engine.search(query, sb_filters, top_k=sb_top_k)

# Query understanding
with st.expander("🧠 What I understood from your query", expanded=False):
    cols = st.columns(6)
    for col, (lbl, val) in zip(cols, [
        ("Bedrooms",  parsed.get("bedrooms") or parsed.get("min_bedrooms")),
        ("Bathrooms", parsed.get("bathrooms")),
        ("Max Rent",  f"£{parsed['max_price']:,}" if parsed.get("max_price")  else None),
        ("Min Rent",  f"£{parsed['min_price']:,}" if parsed.get("min_price")  else None),
        ("Type",      parsed.get("property_type")),
        ("New Build", "Yes" if parsed.get("new_home") else None),
    ]):
        col.metric(lbl, str(val) if val is not None else "—")

# ── Analytical answer ─────────────────────────────────────────────────────────
if analytical_answer:
    st.markdown("### 📊 Market Analysis")
    # Escape the content but render markdown-style bold via replace
    safe = e(analytical_answer).replace("**", "<b>", 1)
    lines_out = []
    for line in analytical_answer.split("\n"):
        safe_line = e(line)
        # restore bold markers
        while "**" in safe_line:
            safe_line = safe_line.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
        lines_out.append(safe_line)
    st.markdown(
        '<div class="analytics-box">' + "<br>".join(lines_out) + "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
if results.empty:
    st.warning("No listings matched. Try widening your price range, "
               "removing a bedroom requirement, or using a different area name.")
    st.stop()

# AI summary
summary = generate_ai_summary(query, results)
st.markdown(
    f'<div class="ai-box">🤖 <strong>AI Summary</strong>&nbsp;&nbsp;{summary}</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown(f"#### {len(results)} listings found")
with right_col:
    view_mode = st.radio("View", ["Cards", "Table"], horizontal=True,
                         label_visibility="collapsed")

# ── Table view ────────────────────────────────────────────────────────────────
if view_mode == "Table":
    disp = results[["neighbourhood", "property_type", "price", "bedrooms",
                     "bathrooms", "crime_score", "flood_risk",
                     "is_new_home", "listing_date"]].copy()
    disp["price"] = disp["price"].apply(lambda x: f"£{int(x):,}/mo")
    disp["is_new_home"] = disp["is_new_home"].map({True: "✅", False: ""})
    disp.columns = ["Area", "Type", "Rent", "Beds", "Baths",
                    "Crime Score", "Flood Risk", "New Build", "Listed"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Card view ─────────────────────────────────────────────────────────────────
else:
    FLOOD_BADGE = {
        "Low":     ("badge-green", "🌿 Low flood risk"),
        "Medium":  ("badge-amber", "🌊 Medium flood risk"),
        "High":    ("badge-red",   "⚠️ High flood risk"),
        "Unknown": ("badge-gray",  "❓ Flood risk unknown"),
    }

    for _, row in results.iterrows():
        crime       = float(row.get("crime_score", 5))
        flood       = str(row.get("flood_risk", "Unknown"))
        is_new      = bool(row.get("is_new_home", False))
        neighbourhood = e(row.get("neighbourhood", "Unknown"))
        prop_type   = e(row.get("property_type", "Property"))
        beds        = int(row.get("bedrooms", 0))
        baths       = int(row.get("bathrooms", 0))
        price       = int(row.get("price", 0))
        listed      = e(row.get("listing_date", ""))
        raw_desc    = str(row.get("description", ""))
        desc        = e(raw_desc[:220] + ("…" if len(raw_desc) > 220 else ""))

        # Crime colour
        if crime >= 7:
            crime_color, crime_label = "#ef4444", "High crime"
        elif crime >= 4:
            crime_color, crime_label = "#f59e0b", "Med crime"
        else:
            crime_color, crime_label = "#22c55e", "Low crime"

        flood_cls, flood_label = FLOOD_BADGE.get(flood, ("badge-gray", flood))

        new_badge = (
            '<span class="badge badge-gold">✨ New Build</span>' if is_new else ""
        )

        st.markdown(f"""
<div class="prop-card">
  <div class="prop-top">
    <div class="prop-left">
      <div class="prop-area">📍 {neighbourhood}</div>
      <div class="prop-badges">
        <span class="badge badge-blue">{prop_type}</span>
        {new_badge}
      </div>
    </div>
    <div style="text-align:right; flex-shrink:0;">
      <div class="prop-price">£{price:,}<span class="prop-price-sub">/mo</span></div>
    </div>
  </div>
  <div class="prop-stats">
    <span class="prop-stat">🛏 {beds} bed</span>
    <span class="prop-stat">🚿 {baths} bath</span>
    <span class="prop-stat">
      <span class="crime-dot" style="background:{crime_color};"></span>{crime:.0f}/10 {crime_label}
    </span>
    <span class="badge {flood_cls}" style="font-size:0.78rem;">{flood_label}</span>
    {"<span class='prop-stat'>📅 " + listed + "</span>" if listed else ""}
  </div>
  <hr class="prop-divider">
  <div class="prop-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
