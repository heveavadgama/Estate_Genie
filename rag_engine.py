"""
Estate Genie — RAG Engine (real dataset build)
Columns: type, bedrooms, bathrooms, price, listing_update_date,
         property_type_full_description, flood_risk, is_new_home,
         laua, crime_score_weight, address
price = monthly rent in GBP; address = area/neighbourhood name.
"""

import re
import numpy as np
import pandas as pd
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

TYPE_NORMALISE = {
    "flat": "Flat", "apartment": "Apartment", "studio": "Studio",
    "terraced": "Terraced", "terrace": "Terraced",
    "semi-detached": "Semi-Detached", "semi detached": "Semi-Detached",
    "detached": "Detached", "house": "House",
    "end of terrace": "End of Terrace", "end-of-terrace": "End of Terrace",
    "maisonette": "Maisonette", "bungalow": "Bungalow",
    "cottage": "Cottage", "villa": "Villa", "penthouse": "Penthouse",
    "link-detached house": "Detached", "town house": "Terraced",
}

PRICE_WORDS = {"k": 1_000, "m": 1_000_000, "thousand": 1_000, "million": 1_000_000}


class QueryParser:
    def parse(self, query: str) -> dict:
        q = query.lower()
        result = {
            "bedrooms": None, "min_bedrooms": None,
            "bathrooms": None,
            "min_price": None, "max_price": None,
            "property_type": None, "neighbourhood": None,
            "new_home": None, "flood_risk": None,
            "is_analytical": False, "sort_by": "relevance",
        }

        analytical_patterns = [
            r"\baverage\b", r"\bmean\b", r"\bmost expensive\b", r"\bcheapest\b",
            r"\bmost affordable\b", r"\bwhich area\b", r"\bcompare\b",
            r"\btop \d+\b", r"\bhighest\b", r"\blowest\b",
            r"\bstatistic\b", r"\banalysis\b", r"\btrend\b",
        ]
        if any(re.search(p, q) for p in analytical_patterns):
            result["is_analytical"] = True

        bed_match = re.search(r"(\d+)\s*(\+)?\s*(?:bed(?:room)?s?|br|bd)\b", q)
        if bed_match:
            n = int(bed_match.group(1))
            if bed_match.group(2):
                result["min_bedrooms"] = n
            else:
                result["bedrooms"] = n

        bath_match = re.search(r"(\d+)\s*(?:\+)?\s*(?:bath(?:room)?s?|ba)\b", q)
        if bath_match:
            result["bathrooms"] = int(bath_match.group(1))

        price_matches = re.findall(r"[£$]\s*([\d,]+)\s*([km]|million|thousand)?", q)
        prices = []
        for num_str, unit in price_matches:
            num = float(num_str.replace(",", ""))
            prices.append(int(num * PRICE_WORDS.get(unit.lower(), 1) if unit else num))
        if not prices:
            plain = re.findall(r"(\d[\d,]*)\s*(?:per\s*month|pcm|pm\b|a\s*month)", q)
            for n in plain:
                prices.append(int(n.replace(",", "")))

        if prices:
            if re.search(r"under|below|less than|cheaper than|max|up to|no more", q):
                result["max_price"] = max(prices)
            elif re.search(r"over|above|more than|at least|min|starting|from", q):
                result["min_price"] = min(prices)
            elif len(prices) >= 2:
                result["min_price"] = min(prices)
                result["max_price"] = max(prices)
            else:
                result["max_price"] = prices[0]

        for raw, norm in TYPE_NORMALISE.items():
            if re.search(rf"\b{re.escape(raw)}s?\b", q):
                result["property_type"] = norm
                break

        if re.search(r"\bnew\s*(?:home|build|property|development)\b", q):
            result["new_home"] = True

        if re.search(r"\bno\s*flood\b|\blow\s*flood\b|\bflood\s*(?:safe|free)\b", q):
            result["flood_risk"] = "Low"

        if re.search(r"most expensive|highest (?:rent|price)|priciest", q):
            result["sort_by"] = "price_desc"
        elif re.search(r"cheapest|lowest (?:rent|price)|most affordable", q):
            result["sort_by"] = "price_asc"

        return result


class EstateGenieRAG:
    def __init__(self, csv_path: str):
        self.df = self._load_and_clean(csv_path)
        self.parser = QueryParser()
        self.model = None
        self.index = None
        self._build_index()

    def _load_and_clean(self, path: str) -> pd.DataFrame:
        import os
        file_size = os.path.getsize(path)
        if file_size < 500:
            # File exists but is tiny — almost certainly a Git LFS pointer
            with open(path) as f:
                snippet = f.read(300)
            raise RuntimeError(
                f"properties.csv is only {file_size} bytes — it looks like a Git LFS "
                f"pointer was committed instead of the real file.\n\n"
                f"File contents:\n{snippet}\n\n"
                f"Fix:\n"
                f"  git lfs untrack '*.csv'\n"
                f"  git rm --cached data/properties.csv\n"
                f"  git add data/properties.csv\n"
                f"  git commit -m 'store csv directly'\n"
                f"  git push"
            )
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read properties.csv ({file_size:,} bytes): {e}\n"
                f"Ensure the file is a valid UTF-8 CSV committed directly to git (not via LFS)."
            ) from e

        df = df.rename(columns={
            "type": "property_type",
            "listing_update_date": "listing_date",
            "property_type_full_description": "description",
            "crime_score_weight": "crime_score",
            "address": "neighbourhood",
        })

        df["property_type"] = (
            df["property_type"].fillna("Other").astype(str).str.lower().str.strip()
            .map(lambda x: TYPE_NORMALISE.get(x, x.title()))
        )

        df = df[df["price"] > 0].copy()
        p999 = df["price"].quantile(0.999)
        df = df[df["price"] <= p999].copy()
        df = df[df["bedrooms"] <= 20].copy()

        def _fr(v):
            v = str(v).strip().lower()
            if v in ("low", "1", "1.0"): return "Low"
            if v in ("medium", "2", "2.0"): return "Medium"
            if v in ("high", "3", "3.0"): return "High"
            return "Unknown"
        df["flood_risk"] = df["flood_risk"].apply(_fr)

        df["listing_date"] = pd.to_datetime(
            df["listing_date"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("")

        df["description"] = df["description"].fillna(
            df["property_type"] + " in " + df["neighbourhood"].fillna("Unknown")
        )
        df["is_new_home"] = df["is_new_home"].fillna(False).astype(bool)
        df["crime_score"] = pd.to_numeric(df["crime_score"], errors="coerce").fillna(5)
        df["neighbourhood"] = df["neighbourhood"].fillna("Unknown").astype(str)

        return df.reset_index(drop=True)

    def _build_index(self):
        if not HAS_FAISS:
            return
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = (
                df["description"].fillna("") + " " +
                df["property_type"].fillna("") + " " +
                df["neighbourhood"].fillna("")
                for df in [self.df]
            )
            texts = list(next(texts))
            embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=128)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        except Exception as e:
            print(f"FAISS build failed: {e}")
            self.index = None

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        if filters.get("bedrooms") is not None:
            df = df[df["bedrooms"] == filters["bedrooms"]]
        if filters.get("min_bedrooms") is not None:
            df = df[df["bedrooms"] >= filters["min_bedrooms"]]
        if filters.get("bathrooms") is not None:
            df = df[df["bathrooms"] >= filters["bathrooms"]]
        if filters.get("min_price") is not None:
            df = df[df["price"] >= filters["min_price"]]
        if filters.get("max_price") is not None:
            df = df[df["price"] <= filters["max_price"]]
        if filters.get("property_type"):
            pt = filters["property_type"].lower()
            df = df[df["property_type"].str.lower() == pt]
        if filters.get("neighbourhood"):
            df = df[df["neighbourhood"].str.lower().str.contains(
                filters["neighbourhood"].lower(), na=False)]
        if filters.get("new_home") is True:
            df = df[df["is_new_home"] == True]
        if filters.get("flood_risk"):
            df = df[df["flood_risk"] == filters["flood_risk"]]
        return df

    def semantic_search(self, query: str, top_k: int = 500) -> pd.DataFrame:
        if self.index is not None and self.model is not None:
            try:
                q_vec = self.model.encode([query], show_progress_bar=False)
                q_vec = np.array(q_vec, dtype="float32")
                faiss.normalize_L2(q_vec)
                scores, indices = self.index.search(q_vec, min(top_k, len(self.df)))
                result = self.df.iloc[indices[0]].copy()
                result["_score"] = scores[0]
                return result
            except Exception:
                pass
        words = query.lower().split()
        mask = self.df["description"].str.lower().apply(
            lambda d: sum(w in str(d) for w in words)
        )
        result = self.df.copy()
        result["_score"] = mask
        return result.sort_values("_score", ascending=False).head(top_k)

    def search(self, query: str, sidebar_filters: Optional[dict] = None, top_k: int = 10):
        filters = self.parser.parse(query)

        if sidebar_filters:
            for k, v in sidebar_filters.items():
                if v is not None and filters.get(k) is None:
                    filters[k] = v

        if filters["is_analytical"]:
            return pd.DataFrame(), filters, self._analytical_answer(query, filters)

        candidates = self.semantic_search(query, top_k=1000)
        filtered = self._apply_filters(candidates, filters)

        if filtered.empty:
            filtered = self._apply_filters(self.df.copy(), filters)

        sort_by = filters.get("sort_by", "relevance")
        if sort_by == "price_desc":
            filtered = filtered.sort_values("price", ascending=False)
        elif sort_by == "price_asc":
            filtered = filtered.sort_values("price", ascending=True)
        elif "_score" in filtered.columns:
            filtered = filtered.sort_values("_score", ascending=False)

        return filtered.head(top_k), filters, ""

    def _analytical_answer(self, query: str, filters: dict) -> str:
        q = query.lower()
        df = self._apply_filters(self.df.copy(), filters)
        if df.empty:
            df = self.df

        lines = []

        if re.search(r"\baverage\b|\bmean\b", q):
            lines.append(f"**Average monthly rent:** £{df['price'].mean():,.0f}")
            lines.append(f"**Median monthly rent:** £{df['price'].median():,.0f}")

        if re.search(r"most expensive|highest rent|priciest", q):
            top = df.nlargest(5, "price")
            lines.append("**Most expensive listings:**")
            for _, r in top.iterrows():
                lines.append(f"  - {r['neighbourhood']}: £{r['price']:,}/mo "
                             f"({r['bedrooms']}bd {r['property_type']})")

        if re.search(r"cheapest|most affordable|lowest rent", q):
            bot = df[df["price"] > 0].nsmallest(5, "price")
            lines.append("**Most affordable listings:**")
            for _, r in bot.iterrows():
                lines.append(f"  - {r['neighbourhood']}: £{r['price']:,}/mo "
                             f"({r['bedrooms']}bd {r['property_type']})")

        if re.search(r"crime|safest|dangerous", q):
            area = df.groupby("neighbourhood")["crime_score"].mean().sort_values(ascending=False)
            lines.append("**Highest crime score areas:**")
            for a, s in area.head(5).items():
                lines.append(f"  - {a}: {s:.1f}/10")
            lines.append("**Safest areas (lowest crime):**")
            for a, s in area.tail(5).items():
                lines.append(f"  - {a}: {s:.1f}/10")

        if re.search(r"compare|vs\.?|versus", q):
            flat_avg = self.df[self.df["property_type"].str.lower().isin(
                ["flat", "apartment"])]["price"].mean()
            house_avg = self.df[self.df["property_type"].str.lower().isin(
                ["terraced", "detached", "semi-detached", "house",
                 "end of terrace"])]["price"].mean()
            lines.append(f"**Flat / Apartment avg rent:** £{flat_avg:,.0f}/mo")
            lines.append(f"**House (all types) avg rent:** £{house_avg:,.0f}/mo")
            diff = house_avg - flat_avg
            lines.append(f"**Difference:** £{abs(diff):,.0f}/mo "
                         f"({'houses more expensive' if diff > 0 else 'flats more expensive'})")

        if re.search(r"top\s+\d+", q):
            m = re.search(r"top\s+(\d+)", q)
            n = int(m.group(1)) if m else 5
            top_n = df.nlargest(n, "price")
            lines.append(f"**Top {n} most expensive:**")
            for _, r in top_n.iterrows():
                lines.append(f"  - {r['neighbourhood']}: £{r['price']:,}/mo "
                             f"({r['bedrooms']}bd {r['property_type']})")

        if re.search(r"which area|best area|popular|most listing", q):
            area_count = df.groupby("neighbourhood").size().sort_values(ascending=False)
            lines.append("**Areas with most listings:**")
            for a, c in area_count.head(8).items():
                lines.append(f"  - {a}: {c:,} listings")

        if not lines:
            lines = [
                f"**Total listings:** {len(self.df):,}",
                f"**Average monthly rent:** £{self.df['price'].mean():,.0f}",
                f"**Median monthly rent:** £{self.df['price'].median():,.0f}",
                f"**Rent range:** £{self.df['price'].min():,} – £{self.df['price'].max():,}/mo",
                f"**Most common type:** {self.df['property_type'].mode()[0]}",
                f"**New build listings:** {self.df['is_new_home'].sum():,}",
            ]

        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {
            "total": len(self.df),
            "avg_price": self.df["price"].mean(),
            "median_price": self.df["price"].median(),
            "min_price": int(self.df["price"].min()),
            "max_price": int(self.df["price"].quantile(0.99)),
            "property_types": sorted(self.df["property_type"].dropna().unique().tolist()),
            "neighbourhoods": sorted(self.df["neighbourhood"].dropna().unique().tolist()),
            "new_home_count": int(self.df["is_new_home"].sum()),
        }
