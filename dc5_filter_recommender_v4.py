
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from math import sqrt, log
from itertools import combinations, combinations_with_replacement
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

def features_dict(seed_digits, prev_digits, prev2_digits):
    s = seed_digits
    total = sum(s)
    spread = (max(s)-min(s)) if s else 0
    if spread <= 3: sb = "0-3"
    elif spread <= 6: sb = "4-6"
    elif spread <= 9: sb = "7-9"
    else: sb = "10+"
    return {
        "sum_cat": sum_category(total),
        "eo": eo_counts(s),
        "hl": hl_counts(s),
        "spread_band": sb,
        "carry_count": len(set(s) & set(prev_digits)),
        "mirror_hit": int(any((MIRROR[d] in prev_digits) for d in set(s))),
        "lead_change": int((len(prev_digits)==5 and s[0] != prev_digits[0])),
        "trail_change": int((len(prev_digits)==5 and s[-1] != prev_digits[-1])),
        "pair_0x": int(any(p in {"00","01","02","03"} for p in pair_strings(s)))
    }

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

    combo_sum = sum(combo_digits)
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
        "combo_sum": combo_sum,
        "seed_sum": sum(seed),
        "prev_sum": sum(prev),
        "prev2_sum": sum(prev2),
        "lead_digit": seed[0] if seed else None,
        "trail_digit": seed[-1] if seed else None,
        "mirror_of": lambda d: MIRROR.get(d, None),
        "is_even": lambda d: d in EVEN,
        "is_odd": lambda d: d in ODD,
        "is_low": lambda d: d in LOW,
        "is_high": lambda d: d in HIGH,
    }
    # Aliases for older CSVs
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
    return loc

def evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, combo_digits):
    try:
        loc = build_context_vars(seed_row, follower_mat, carry_rates, combo_digits)
        # If the CSV accidentally wraps the expression in quotes, eval once to strip
        if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
            expr = eval(expr)
        res = safe_eval(expr, loc)
        eliminated = bool(res)
        return not eliminated
    except Exception:
        return True

def winner_kept_by_filter(expr, seed_row, follower_mat, carry_rates):
    w = seed_row["winner_digits"]
    return evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, w)

# ========================
# File parsing & order controls
# ========================

def parse_history_upload(file):
    name = file.name.lower()
    content = file.getvalue()  # bytes
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        text = content.decode("utf-8", errors="ignore")
        lines = text.strip().splitlines()
        dates, results = [], []
        for line in lines:
            # First try strict split
            parts = re.split(r"\\t+|\\s{2,}", line.strip())
            parts = [p for p in parts if p]
            if len(parts) >= 2 and re.fullmatch(r"\\d{5}", parts[-1]):
                dates.append(parts[0]); results.append(parts[-1]); continue
            # Fallback: last 5-digit token anywhere
            m = re.search(r"(\\d{5})(?!.*\\d)", line)
            if m:
                dates.append(line[:m.start()].strip())
                results.append(m.group(1))
        df = pd.DataFrame({"Date": dates, "Result": results})
    # Normalize
    if "Result" not in df.columns:
        cand = [c for c in df.columns if str(c).lower().strip() in {"result","results","number","draw","winner"}]
        if cand:
            df["Result"] = df[cand[0]]
        else:
            raise ValueError("Could not find a 'Result' column.")
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            try:
                df["Date"] = pd.to_datetime(df["Date"].str.replace(r"^[A-Za-z]{3},\\s*", "", regex=True))
            except Exception:
                pass
    return df

# ========================
# Streamlit UI
# ========================

st.set_page_config(page_title="DC-5 Seed-Aware Filter Recommender v4", layout="wide")
st.title("DC-5 Seed-Aware Filter Recommender — Robust Parse + Diagnostics (v4)")

with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Draw history (CSV or TXT)", type=["csv","txt"])
    filt_file = st.file_uploader("Filter CSV (id/name/enabled/applicable_if/expression)", type=["csv"])
    hot_win = st.number_input("Hot/Cold window (draws)", min_value=5, max_value=100, value=20, step=1)
    min_bucket_n = st.number_input("Min bucket sample size", min_value=5, max_value=200, value=20, step=1)
    keep_lb_thresh = st.slider("Min Wilson-LB KeepRate for SAFE", 0.50, 1.00, 0.90, 0.01)
    elim_min = st.slider("Min elimination fraction for scoring", 0.0, 0.8, 0.20, 0.01)
    current_idx = st.number_input("Target seed index (0 = earliest after ordering)", min_value=0, value=0, step=1)
    flip_reverse = st.checkbox("Flip if file is reverse-chron (newest→oldest)", value=True)
    min_signal_n = st.number_input("Diagnostics: min support per cell", min_value=5, max_value=200, value=20, step=1)
    run_btn = st.button("Run")

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
        st.stop()

    # Parse files
    try:
        hist_df = parse_history_upload(hist_file)
    except Exception as e:
        st.error(f"Could not parse history file: {e}")
        st.stop()

    try:
        filt_df_raw = pd.read_csv(filt_file)
    except Exception as e:
        st.error(f"Could not read filter CSV: {e}")
        st.stop()

    filt_df = normalize_filter_df(filt_df_raw)
    if filt_df.empty:
        st.error("No valid filters with expressions found in the filter CSV.")
        st.stop()

    # Order
    if "Date" in hist_df.columns and pd.api.types.is_datetime64_any_dtype(hist_df["Date"]):
        # Detect reverse by comparing first and last
        asc = hist_df["Date"].is_monotonic_increasing
        desc = hist_df["Date"].is_monotonic_decreasing
        if desc and flip_reverse:
            hist_df = hist_df.sort_values("Date", ascending=True).reset_index(drop=True)
        elif not asc:
            hist_df = hist_df.sort_values("Date", ascending=True).reset_index(drop=True)
    else:
        if flip_reverse:
            hist_df = hist_df.iloc[::-1].reset_index(drop=True)

    # Preview
    st.subheader("History sanity check")
    st.write(f"Rows found: **{len(hist_df)}**")
    if "Date" in hist_df.columns:
        st.write("Earliest → Latest:", hist_df["Date"].head(1).iloc[0], "→", hist_df["Date"].tail(1).iloc[0])
    st.write("Head:")
    st.dataframe(hist_df.head(5), use_container_width=True)
    st.write("Tail:")
    st.dataframe(hist_df.tail(5), use_container_width=True)

    if "Result" not in hist_df.columns or len(hist_df) < 6:
        st.error("Need at least 6 rows with a 'Result' column to compute walk-forward stats.")
        st.stop()

    # Build rows (time-ordered)
    results = hist_df["Result"].astype(str).tolist()
    rows = []
    for i in range(len(results)-1):
        seed = to_digits(results[i]); win = to_digits(results[i+1])
        prev = to_digits(results[i-1]) if i-1 >= 0 else []
        prev2 = to_digits(results[i-2]) if i-2 >= 0 else []

        start_idx = max(0, i-int(hot_win))
        window_vals = [d for s in results[start_idx:i] for d in to_digits(s)]
        cnt = Counter(window_vals)
        hot_sorted = [d for d,_ in cnt.most_common()]
        cold_sorted = [d for d,_ in cnt.most_common()[::-1]]
        hot6 = hot_sorted[:6] if hot_sorted else []
        hot7 = hot_sorted[:7] if hot_sorted else []
        cold6 = cold_sorted[:6] if cold_sorted else []
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

    # Build prior-only keep cache
    def follower_matrix_upto(idx):
        return follower_matrix([{"seed_digits": rr["seed_digits"], "winner_digits": rr["winner_digits"]} for rr in rows[:idx]])
    def carry_rates_upto(idx):
        return carry_rate_map([(rr["seed_digits"], rr["winner_digits"]) for rr in rows[:idx]])

    per_index_keep = defaultdict(dict)
    for i in range(len(rows)):
        if i>0:
            follower_mat = follower_matrix_upto(i)
            carry_rates = carry_rates_upto(i)
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
                    aexpr = app
                    if (aexpr.startswith('"') and aexpr.endswith('"')) or (aexpr.startswith("'") and aexpr.endswith("'")):
                        aexpr = eval(aexpr)
                    applicable = bool(safe_eval(aexpr, build_context_vars(r, follower_mat, carry_rates, r["seed_digits"])))
                except Exception:
                    applicable = True
            if not applicable:
                per_index_keep[i][f["id"]] = None
                continue
            kept = winner_kept_by_filter(f["expression"], r, follower_mat, carry_rates)
            per_index_keep[i][f["id"]] = int(kept)

    # Current seed
    cur = int(current_idx)
    if cur < 0 or cur >= len(rows):
        st.error(f"Target seed index out of range. Valid 0..{len(rows)-1}.")
        st.stop()

    rcur = rows[cur]
    pool_boxes = step0_generate_pool(rcur["seed_digits"])

    # Bucket stats from prior rows only
    def bkey(r):
        return tuple(features_dict(r["seed_digits"], r["prev_digits"], r["prev2_digits"]).items())

    filter_stats_bucket = defaultdict(lambda: {"kept":0, "total":0})
    for i in range(cur):
        r = rows[i]
        key = bkey(r)
        for _,f in filt_df.iterrows():
            if not f["enabled"]:
                continue
            val = per_index_keep[i].get(f["id"])
            if val is None:
                continue
            fb = (f["id"], key)
            filter_stats_bucket[fb]["total"] += 1
            filter_stats_bucket[fb]["kept"] += int(val==1)

    # Score for current seed
    records = []
    if cur>0:
        follower_mat_cur = follower_matrix_upto(cur)
        carry_rates_cur = carry_rates_upto(cur)
    else:
        follower_mat_cur = {d:Counter() for d in DIGITS}
        carry_rates_cur = {d:0.0 for d in DIGITS}

    for _,f in filt_df.iterrows():
        if not f["enabled"]:
            continue
        fid = f["id"]; fname = f["name"]; expr = f["expression"]
        # applicable?
        app = f["applicable_if"].strip()
        applicable = True
        if app:
            try:
                aexpr = app
                if (aexpr.startswith('"') and aexpr.endswith('"')) or (aexpr.startswith("'") and aexpr.endswith("'")):
                    aexpr = eval(aexpr)
                applicable = bool(safe_eval(aexpr, build_context_vars(rcur, follower_mat_cur, carry_rates_cur, rcur["seed_digits"])))
            except Exception:
                applicable = True
        if not applicable:
            continue
        key_cur = bkey(rcur)
        kept_b = filter_stats_bucket.get((fid, key_cur), {"kept":0,"total":0})
        kept_bucket, total_bucket = kept_b["kept"], kept_b["total"]
        lb_bucket = wilson_lower_bound(kept_bucket, total_bucket, z=1.96)

        # elim frac
        survive = 0
        for box in pool_boxes:
            digs = [int(c) for c in box]
            if evaluate_filter_on_combo(expr, rcur, follower_mat_cur, carry_rates_cur, digs):
                survive += 1
        elim_frac = (len(pool_boxes)-survive)/len(pool_boxes) if pool_boxes else 0.0
        score = lb_bucket * log(1 + max(0.0, elim_frac))
        tag = "Safe" if (lb_bucket >= keep_lb_thresh and elim_frac >= elim_min and total_bucket >= min_bucket_n) else ("Watch" if lb_bucket >= 0.80 else "Risky")
        role = "LoserList" if str(fid).strip().upper().startswith("LL") else ""
        records.append({
            "filter_id": fid,
            "name": fname,
            "Role": role,
            "keep_kept": kept_bucket,
            "keep_total": total_bucket,
            "keep_LB": round(lb_bucket, 3),
            "elim_frac_current": round(elim_frac, 3),
            "elim_count_current": int(round(elim_frac*len(pool_boxes))) if pool_boxes else 0,
            "pool_size": len(pool_boxes),
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
        st.write("Carry-over count:", len(set(rcur["seed_digits"]) & set(rcur["prev_digits"])))
    with c2:
        st.write("Sum category:", sum_category(sum(rcur["seed_digits"])))
        st.write("EO:", eo_counts(rcur["seed_digits"]))
        st.write("HL:", hl_counts(rcur["seed_digits"])
)
    with c3:
        spread = (max(rcur["seed_digits"])-min(rcur["seed_digits"])) if rcur["seed_digits"] else 0
        st.write("Spread:", spread)
        st.write("Step-0 pool size:", len(pool_boxes))

    st.subheader("Recommended filters (scored, prior-only)")
    st.dataframe(rec_df, use_container_width=True)

    # Diagnostics: per-feature safety table for a picked filter
    st.markdown("### Diagnostics: Seed characteristics → Filter safety")
    if len(rec_df) == 0:
        st.info("No applicable filters to analyze.")
        st.stop()

    pick = st.selectbox("Pick a filter to analyze", options=[f"{r['filter_id']} — {r['name']}" for _,r in rec_df.iterrows()], index=0)
    pick_id = pick.split(" — ")[0]

    # Build feature dataframe for prior rows
    feat_rows = []
    for i,r in enumerate(rows):
        fdict = features_dict(r["seed_digits"], r["prev_digits"], r["prev2_digits"])
        fdict["i"] = i
        feat_rows.append(fdict)
    feat_df = pd.DataFrame(feat_rows)

    feat_cols = ["sum_cat","eo","hl","spread_band","carry_count","mirror_hit","lead_change","trail_change","pair_0x"]
    rows_prior = list(range(cur))
    agg = defaultdict(lambda: {"kept":0, "total":0})
    for feat in feat_cols:
        for i in rows_prior:
            val = feat_df.loc[i, feat]
            keep_val = per_index_keep[i].get(pick_id)
            if keep_val is None: continue
            agg[(feat, val)]["kept"] += int(keep_val==1)
            agg[(feat, val)]["total"] += 1

    rows_out = []
    for (feat, val), kt in agg.items():
        kept = kt["kept"]; total = kt["total"]
        if total < int(min_signal_n): continue
        lb = wilson_lower_bound(kept, total, z=1.96)
        rows_out.append({
            "feature": feat,
            "value": val,
            "kept": kept,
            "total": total,
            "keep_LB": round(lb,3),
            "coverage_%": round(100*total/max(1, len(rows_prior)),1),
            "matches_current": bool(val == feat_df.loc[cur, feat])
        })
    diag_df = pd.DataFrame(rows_out).sort_values(["matches_current","keep_LB","total"], ascending=[False, False, False]).reset_index(drop=True)

    st.dataframe(diag_df, use_container_width=True)

    csv_buf = io.StringIO()
    rec_df.to_csv(csv_buf, index=False)
    st.download_button("Download recommendations CSV", data=csv_buf.getvalue(), file_name="filter_recommendations_v4.csv", mime="text/csv")
