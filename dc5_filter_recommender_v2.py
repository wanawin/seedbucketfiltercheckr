
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import sqrt, log
from itertools import combinations, combinations_with_replacement, product
import io
import re

# ========================
# Utility helpers
# ========================

DIGITS = list(range(10))
EVEN = {0,2,4,6,8}
ODD = {1,3,5,7,9}
LOW = {0,1,2,3,4}
HIGH = {5,6,7,8,9}
MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

def to_digits(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    ds = [int(c) for c in s if c.isdigit()]
    # pad/truncate to 5 if needed (defensive)
    if len(ds) > 5:
        ds = ds[-5:]
    elif len(ds) < 5:
        ds = ([0] * (5 - len(ds))) + ds
    return ds

def sort_box_str(digs):
    return ''.join(map(str, sorted(digs)))

def eo_counts(digs):
    e = sum(1 for d in digs if d in EVEN)
    o = len(digs) - e
    return f"{e}E/{o}O"

def hl_counts(digs):
    h = sum(1 for d in digs if d in HIGH)
    l = len(digs) - h
    return f"{h}H/{l}L"

def sum_category(total):
    if total <= 15: return "VeryLow"
    if total <= 24: return "Low"
    if total <= 33: return "Mid"
    return "High"  # 34–45

def wilson_lower_bound(k, n, z=1.96):
    if n == 0: return 0.0
    phat = k / n
    denom = 1 + z**2/n
    centre = phat + z**2/(2*n)
    adj = z * ((phat*(1-phat) + z**2/(4*n)) / n) ** 0.5
    return (centre - adj) / denom

def safe_eval(expr, local_vars):
    allowed_builtins = {
        "sum": sum, "any": any, "all": all, "len": len, "min": min, "max": max, "sorted": sorted, "set": set,
        "list": list, "tuple": tuple, "abs": abs, "range": range, "map": map, "filter": filter, "enumerate": enumerate,
        "zip": zip, "Counter": Counter
    }
    return eval(expr, {"__builtins__": {}}, {**allowed_builtins, **local_vars})

# ========================
# Feature computation
# ========================

def follower_matrix(rows):
    mat = {d: Counter() for d in DIGITS}
    for r in rows:
        for d in set(r["seed_digits"]):
            for y in r["winner_digits"]:
                mat[d][y] += 1
    return mat

def follower_pool_for_seed(seed_digits, mat, top_n=6):
    cnt = Counter()
    for d in set(seed_digits):
        cnt.update(mat[d])
    if not cnt:
        return []
    return [d for d,_ in cnt.most_common(top_n)]

def carry_rate_map(train_pairs):
    seen = Counter(); carried = Counter()
    for seed, win in train_pairs:
        for d in set(seed):
            seen[d] += 1
            if d in win:
                carried[d] += 1
    rates = {d: (carried[d]/seen[d]) if seen[d] else 0.0 for d in DIGITS}
    return rates

def top2_carry_candidates(prev_digits, carry_rates):
    if not prev_digits: return []
    digs = list(set(prev_digits))
    digs.sort(key=lambda d: (-carry_rates.get(d,0.0), d))
    return digs[:2]

def build_rows_from_history(df, hot_window=20):
    results = df["Result"].astype(str).tolist()
    rows = []
    for i in range(len(results)-1):
        seed = to_digits(results[i])
        win = to_digits(results[i+1])
        prev = to_digits(results[i-1]) if i-1 >= 0 else []
        prev2 = to_digits(results[i-2]) if i-2 >= 0 else []

        # rolling window excludes the current seed row
        start_idx = max(0, i-hot_window)
        window_vals = [d for s in results[start_idx:i] for d in to_digits(s)]
        cnt = Counter(window_vals)
        hot_sorted = [d for d,_ in cnt.most_common()]
        cold_sorted = [d for d,_ in cnt.most_common()[::-1]]
        hot6 = hot_sorted[:6] if hot_sorted else []
        hot7 = hot_sorted[:7] if hot_sorted else []
        cold6 = cold_sorted[:6] if cold_sorted else []

        # due last 2
        due_last2 = sorted(set(DIGITS) - set(prev) - set(prev2))

        s2 = sorted({(d+1) % 10 for d in seed})
        union_digits = sorted(set(seed) | set(s2))

        rows.append({
            "i": i,
            "seed_digits": seed,
            "winner_digits": win,
            "prev_digits": prev,
            "prev2_digits": prev2,
            "hot_last20": hot6,
            "hot7_last20": hot7,
            "cold_last20": cold6,
            "due_last2": due_last2,
            "s2": s2,
            "UNION_DIGITS": union_digits,
        })
    return rows

def pair_strings(digs):
    c = Counter(digs)
    pairs = set()
    for d, cnt in c.items():
        if cnt >= 2:
            pairs.add(f"{d}{d}")
    uniq = sorted(c.keys())
    for a,b in combinations(uniq, 2):
        pairs.add(f"{min(a,b)}{max(a,b)}")
    return pairs

def bucket_key(seed_digits, prev_digits, prev2_digits):
    s = seed_digits
    total = sum(s)
    key = {
        "sum_cat": sum_category(total),
        "eo": eo_counts(s),
        "hl": hl_counts(s),
        "spread": max(s)-min(s) if s else 0,
        "carry_count": len(set(s) & set(prev_digits)),
        "mirror_hit": int(any((MIRROR[d] in prev_digits) for d in set(s))),
        "lead_change": int((len(prev_digits)==5 and s[0] != prev_digits[0])),
        "trail_change": int((len(prev_digits)==5 and s[-1] != prev_digits[-1])),
        "pair_0x": int(any(p in {"00","01","02","03"} for p in pair_strings(s)))
    }
    return tuple(key.items())

# ========================
# Step 0 Generator
# ========================

def step0_generate_pool(seed_digits):
    sd = seed_digits
    c = Counter(sd)
    seed_pairs = set()
    for d, cnt in c.items():
        if cnt >= 2:
            seed_pairs.add(tuple(sorted([d,d])))
    uniq = sorted(c.keys())
    for a,b in combinations(uniq, 2):
        seed_pairs.add(tuple(sorted([a,b])))
    if not seed_pairs:
        return []
    boxes = set()
    for x,y,z in combinations_with_replacement(DIGITS, 3):
        add3 = [x,y,z]
        for p in seed_pairs:
            combo = list(p) + add3
            cnts = Counter(combo)
            if all(v <= 2 for v in cnts.values()):
                boxes.add(sort_box_str(combo))
    return sorted(boxes)

# ========================
# Filter evaluation
# ========================

def build_context_vars(seed_row, follower_mat, carry_rates, combo_digits):
    seed = seed_row["seed_digits"]
    prev = seed_row["prev_digits"]
    prev2 = seed_row["prev2_digits"]

    follower_top6 = follower_pool_for_seed(seed, follower_mat, top_n=6)
    follower_top2 = follower_pool_for_seed(seed, follower_mat, top_n=2)

    c2 = top2_carry_candidates(prev, carry_rates)
    u2 = sorted(set(c2) | set(seed_row["s2"]))

    loc = {
        "DIGITS": DIGITS,
        "EVEN": EVEN, "ODD": ODD, "LOW": LOW, "HIGH": HIGH,
        "MIRROR": MIRROR,
        "seed_digits": seed,
        "prev_digits": prev,
        "prev2_digits": prev2,
        "hot_last20": seed_row["hot_last20"],
        "hot7_last20": seed_row["hot7_last20"],
        "cold_last20": seed_row["cold_last20"],
        "due_last2": seed_row["due_last2"],
        "s2": seed_row["s2"],
        "UNION_DIGITS": seed_row["UNION_DIGITS"],
        "c2": c2,
        "u2": u2,
        "follower_top6": follower_top6,
        "follower_top2": follower_top2,
        "combo_digits": combo_digits,
        "mirror_of": lambda d: MIRROR.get(d, None),
        "is_even": lambda d: d in EVEN,
        "is_odd": lambda d: d in ODD,
        "is_low": lambda d: d in LOW,
        "is_high": lambda d: d in HIGH,
    }

    # ---- Compatibility aliases for older filter CSVs ----
    loc["hot7"] = loc["hot7_last20"]
    loc["hot6"] = loc["hot_last20"]
    loc["hot"] = loc["hot_last20"]
    loc["cold"] = loc["cold_last20"]
    loc["due2"] = loc["due_last2"]
    loc["union"] = loc["UNION_DIGITS"]
    loc["Union"] = loc["UNION_DIGITS"]
    loc["UNION"] = loc["UNION_DIGITS"]
    loc["seed_plus_1"] = loc["s2"]
    loc["carry2"] = loc["c2"]
    loc["carry_top2"] = loc["c2"]
    # -----------------------------------------------------

    return loc

def evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, combo_digits):
    try:
        loc = build_context_vars(seed_row, follower_mat, carry_rates, combo_digits)
        res = safe_eval(expr, loc)
        eliminated = bool(res)
        return not eliminated
    except Exception:
        return True

def winner_kept_by_filter(expr, seed_row, follower_mat, carry_rates):
    w = seed_row["winner_digits"]
    return evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, w)

def elimination_fraction(expr, seed_row, follower_mat, carry_rates, pool_boxes):
    if not pool_boxes:
        return 0.0, 0, 0
    survive = 0
    for box in pool_boxes:
        digs = [int(c) for c in box]
        if evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, digs):
            survive += 1
    elim = len(pool_boxes) - survive
    return (elim/len(pool_boxes)), int(elim), len(pool_boxes)

# ========================
# kNN (simple Hamming on discrete features)
# ========================

def seed_vector(seed_row):
    s = seed_row["seed_digits"]
    prev = seed_row["prev_digits"]
    total = sum(s)
    vec = []
    vec.append({"VeryLow":0,"Low":1,"Mid":2,"High":3}[sum_category(total)])
    e = sum(1 for d in s if d in EVEN); vec.append(e)
    h = sum(1 for d in s if d in HIGH); vec.append(h)
    vec.append(max(s)-min(s) if s else 0)
    vec.append(len(set(s)&set(prev)))
    vec.append(int(any(MIRROR[d] in prev for d in set(s))))
    vec.append(int((len(prev)==5 and s[0] != prev[0])))
    vec.append(int((len(prev)==5 and s[-1] != prev[-1])))
    vec.append(int(any(p in {"00","01","02","03"} for p in pair_strings(s))))
    return tuple(vec)

def hamming(a,b):
    return sum(int(x!=y) for x,y in zip(a,b))

def knn_indices(vectors, idx, k=50):
    target = vectors[idx]
    dists = []
    for j,v in enumerate(vectors[:idx]):
        d = hamming(target, v)
        dists.append((d,j))
    dists.sort(key=lambda t: (t[0], t[1]))
    return [j for _,j in dists[:k]]

# ========================
# File parsing & order controls
# ========================

def parse_history_upload(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        # Try to parse a 2-column TXT: Date + Result
        # Split by two or more spaces or tabs
        lines = file.read().decode("utf-8", errors="ignore").strip().splitlines()
        dates = []
        results = []
        for line in lines:
            parts = re.split(r"\\t+|\\s{2,}", line.strip())
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                date_str = parts[0]
                res_str = parts[-1]
                dates.append(date_str)
                results.append(res_str)
        df = pd.DataFrame({"Date": dates, "Result": results})
    # Normalize columns
    if "Result" not in df.columns:
        # Try to find the result column
        cand = [c for c in df.columns if str(c).lower().strip() in {"result","results","number","draw","winner"}]
        if cand:
            df["Result"] = df[cand[0]]
        else:
            raise ValueError("Could not find a 'Result' column.")
    # Coerce Date if present
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            # Try stripping weekday if present
            try:
                df["Date"] = pd.to_datetime(df["Date"].str.replace(r"^[A-Za-z]{3},\\s*", "", regex=True))
            except Exception:
                pass
    return df

def detect_reverse(df):
    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        # True if strictly descending
        d = df["Date"].dropna().values
        if len(d) >= 2:
            return bool((pd.Series(d).diff().dt.total_seconds().fillna(0) < 0).all())
    return False

# ========================
# Streamlit UI
# ========================

st.set_page_config(page_title="DC-5 Seed-Aware Filter Recommender v2", layout="wide")

st.title("DC-5 Seed-Aware Filter Recommender — Chronology-Safe (v2)")
st.caption("Handles reverse-chronology files, older filter variable names, and tags Loser List filters.")

with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Draw history (CSV or TXT)", type=["csv","txt"])
    filt_file = st.file_uploader("Filter CSV (id/name/enabled/applicable_if/expression)", type=["csv"])
    hot_win = st.number_input("Hot/Cold window (draws)", min_value=5, max_value=100, value=20, step=1)
    k_neighbors = st.number_input("kNN size (k)", min_value=5, max_value=200, value=50, step=5)
    min_bucket_n = st.number_input("Min bucket sample size", min_value=5, max_value=200, value=20, step=1)
    keep_lb_thresh = st.slider("Min Wilson-LB KeepRate for SAFE", 0.50, 1.00, 0.90, 0.01)
    elim_min = st.slider("Min elimination fraction for scoring", 0.0, 0.8, 0.20, 0.01)
    current_idx = st.number_input("Target seed index (0 = earliest after ordering)", min_value=0, value=0, step=1)
    run_btn = st.button("Run Recommender")

def normalize_filter_df(df):
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if "id" not in df.columns and "ID" in df.columns:
        df["id"] = df["ID"]
    if "name" not in df.columns and "Name" in df.columns:
        df["name"] = df["Name"]
    if "enabled" not in df.columns:
        if "Enabled" in df.columns:
            df["enabled"] = df["Enabled"]
        else:
            df["enabled"] = True
    if "applicable_if" not in df.columns:
        df["applicable_if"] = ""
    if "expression" not in df.columns and "Expression" in df.columns:
        df["expression"] = df["Expression"]
    df["id"] = df["id"].astype(str)
    df["name"] = df.get("name", df["id"]).astype(str)
    def parse_bool(x):
        if isinstance(x, bool): return x
        s = str(x).strip().lower()
        return s not in {"false","0","no","off",""}
    df["enabled"] = df["enabled"].map(parse_bool)
    df["applicable_if"] = df["applicable_if"].fillna("").astype(str)
    df["expression"] = df["expression"].fillna("").astype(str)
    df = df[df["expression"].str.len()>0].copy()
    return df[["id","name","enabled","applicable_if","expression"]]

if run_btn:
    if not hist_file or not filt_file:
        st.error("Please provide both the history file and the filter CSV.")
    else:
        try:
            hist_df = parse_history_upload(hist_file)
        except Exception as e:
            st.error(f"Could not parse history file: {e}")
            st.stop()

        # Chronology controls
        auto_rev = detect_reverse(hist_df)
        col_a, col_b = st.columns(2)
        with col_a:
            if "Date" in hist_df.columns:
                st.write("Detected Date column.")
                st.write("Sample:", hist_df[["Date","Result"]].head(3))
            else:
                st.write("No Date column detected; using file row order.")
        with col_b:
            reverse_flag = st.checkbox("File is reverse-chronology (newest → oldest). Flip to chronological.", value=auto_rev)

        # Order the dataframe
        if "Date" in hist_df.columns and pd.api.types.is_datetime64_any_dtype(hist_df["Date"]):
            # If reverse flag, sort ascending, else keep as-is (assuming already ascending)
            if reverse_flag:
                hist_df = hist_df.sort_values("Date").reset_index(drop=True)
            else:
                # ensure ascending; if already ascending, this is a no-op
                hist_df = hist_df.sort_values("Date").reset_index(drop=True)
        else:
            # No Date column; just reverse by rows if flagged
            if reverse_flag:
                hist_df = hist_df.iloc[::-1].reset_index(drop=True)

        if "Result" not in hist_df.columns:
            st.error("History must include a 'Result' column after parsing.")
            st.stop()

        # Build rows
        rows = build_rows_from_history(hist_df, hot_window=int(hot_win))
        if len(rows) < 5:
            st.error("Not enough rows to compute walk-forward stats.")
            st.stop()

        filt_df_raw = pd.read_csv(filt_file)
        filt_df = normalize_filter_df(filt_df_raw)
        if filt_df.empty:
            st.error("No valid filters with expressions found in the filter CSV.")
            st.stop()

        # Precompute vectors for kNN
        vectors = [seed_vector(r) for r in rows]

        # Precompute per-index keep flags using *prior* rows only
        per_index_keep = defaultdict(dict)
        for i in range(len(rows)):
            # follower & carry from prior rows only
            if i>0:
                follower_mat = follower_matrix([{"seed_digits": rr["seed_digits"], "winner_digits": rr["winner_digits"]} for rr in rows[:i]])
                carry_rates = carry_rate_map([(rr["seed_digits"], rr["winner_digits"]) for rr in rows[:i]])
            else:
                follower_mat = {d:Counter() for d in DIGITS}
                carry_rates = {d:0.0 for d in DIGITS}
            r = rows[i]
            for _,f in filt_df.iterrows():
                if not f["enabled"]:
                    continue
                app = f["applicable_if"].strip()
                applicable = True
                if app:
                    try:
                        applicable = bool(safe_eval(app, build_context_vars(r, follower_mat, carry_rates, r["seed_digits"])))
                    except Exception:
                        applicable = True
                if not applicable:
                    per_index_keep[i][f["id"]] = None
                    continue
                kept = winner_kept_by_filter(f["expression"], r, follower_mat, carry_rates)
                per_index_keep[i][f["id"]] = int(kept)

        # Aggregate bucket stats walk-forward
        filter_stats_bucket = defaultdict(lambda: {"kept":0, "total":0})
        for i in range(len(rows)):
            r = rows[i]
            bkey = bucket_key(r["seed_digits"], r["prev_digits"], r["prev2_digits"])
            for _,f in filt_df.iterrows():
                if not f["enabled"]:
                    continue
                val = per_index_keep[i].get(f["id"])
                if val is None:
                    continue
                fb = (f["id"], bkey)
                filter_stats_bucket[fb]["total"] += 1
                filter_stats_bucket[fb]["kept"] += int(val==1)

        # Build kNN stats per index
        knn_stats_by_index = defaultdict(dict)
        for i in range(len(rows)):
            idxs = knn_indices(vectors, i, k=int(k_neighbors))
            for _,f in filt_df.iterrows():
                if not f["enabled"]:
                    continue
                kept = 0; total = 0
                for j in idxs:
                    val = per_index_keep[j].get(f["id"])
                    if val is None:
                        continue
                    total += 1
                    kept += int(val==1)
                knn_stats_by_index[i][f["id"]] = (kept, total)

        # Current seed
        cur = int(current_idx)
        if cur < 0 or cur >= len(rows):
            st.error(f"Target seed index out of range. Valid 0..{len(rows)-1}.")
            st.stop()

        if cur>0:
            follower_mat_cur = follower_matrix([{"seed_digits": rr["seed_digits"], "winner_digits": rr["winner_digits"]} for rr in rows[:cur]])
            carry_rates_cur = carry_rate_map([(rr["seed_digits"], rr["winner_digits"]) for rr in rows[:cur]])
        else:
            follower_mat_cur = {d:Counter() for d in DIGITS}
            carry_rates_cur = {d:0.0 for d in DIGITS}

        rcur = rows[cur]
        bkey_cur = bucket_key(rcur["seed_digits"], rcur["prev_digits"], rcur["prev2_digits"])
        pool_boxes = step0_generate_pool(rcur["seed_digits"])

        # Score
        records = []
        for _,f in filt_df.iterrows():
            if not f["enabled"]:
                continue
            fid = f["id"]; fname = f["name"]; expr = f["expression"]
            app = f["applicable_if"].strip()
            applicable = True
            if app:
                try:
                    applicable = bool(safe_eval(app, build_context_vars(rcur, follower_mat_cur, carry_rates_cur, rcur["seed_digits"])))
                except Exception:
                    applicable = True
            if not applicable:
                continue
            kept_b = filter_stats_bucket.get((fid, bkey_cur), {"kept":0,"total":0})
            kept_bucket, total_bucket = kept_b["kept"], kept_b["total"]
            lb_bucket = wilson_lower_bound(kept_bucket, total_bucket, z=1.96)
            kept_knn, total_knn = knn_stats_by_index[cur].get(fid, (0,0))
            lb_knn = wilson_lower_bound(kept_knn, total_knn, z=1.96)
            lb = max(lb_bucket, lb_knn)
            elim_frac, elim_count, pool_size = elimination_fraction(expr, rcur, follower_mat_cur, carry_rates_cur, pool_boxes)
            score = lb * log(1 + max(0.0, elim_frac))
            role = "LoserList" if str(fid).strip().upper().startswith("LL") else ""
            tag = "Safe" if (lb >= keep_lb_thresh and elim_frac >= elim_min and total_bucket >= min_bucket_n) else ("Watch" if lb >= 0.80 else "Risky")
            records.append({
                "filter_id": fid,
                "name": fname,
                "Role": role,
                "keep_kept": kept_bucket,
                "keep_total": total_bucket,
                "keep_LB": round(lb, 3),
                "kNN_kept": kept_knn,
                "kNN_total": total_knn,
                "kNN_LB": round(lb_knn, 3),
                "elim_frac_current": round(elim_frac, 3),
                "elim_count_current": int(elim_count),
                "pool_size": pool_size,
                "Score": round(score, 6),
                "Tag": tag,
            })

        rec_df = pd.DataFrame(records).sort_values(["Tag","Score","keep_LB"], ascending=[True, False, False]).reset_index(drop=True)

        st.subheader("Seed context (current)")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.write("Seed digits:", rcur["seed_digits"])
            st.write("Prev digits:", rcur["prev_digits"])
            st.write("Prev2 digits:", rcur["prev2_digits"])
        with c2:
            st.write("Hot (last 20):", rcur["hot_last20"])
            st.write("Cold (last 20):", rcur["cold_last20"])
            st.write("Due (last 2):", rcur["due_last2"])
        with c3:
            st.write("s2 (seed+1):", rcur["s2"])
            st.write("UNION_DIGITS:", rcur["UNION_DIGITS"])
            st.write("Step-0 pool size:", len(pool_boxes))

        st.subheader("Recommended filters (scored)")
        st.dataframe(rec_df, use_container_width=True)

        safe_df = rec_df[rec_df["Tag"]=="Safe"].sort_values("Score", ascending=False)
        lean_df = safe_df.head(3)
        risky_df = rec_df[(rec_df["Tag"]!="Safe") & (rec_df["keep_LB"]>=0.80)].sort_values("Score", ascending=False).head(5)

        st.markdown("**Core Safe Set (all 'Safe')** — ordered by Score")
        st.dataframe(safe_df[["filter_id","name","Role","keep_LB","elim_frac_current","Score"]], use_container_width=True)

        st.markdown("**Lean Set (top 3 from Safe)**")
        st.dataframe(lean_df[["filter_id","name","Role","keep_LB","elim_frac_current","Score"]], use_container_width=True)

        st.markdown("**Aggressive Adds (top 'Watch' / borderline)**")
        st.dataframe(risky_df[["filter_id","name","Role","kNN_LB","elim_frac_current","Score"]], use_container_width=True)

        csv_buf = io.StringIO()
        rec_df.to_csv(csv_buf, index=False)
        st.download_button("Download full recommendations CSV", data=csv_buf.getvalue(), file_name="filter_recommendations_v2.csv", mime="text/csv")
