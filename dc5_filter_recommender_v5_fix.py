
import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement
from math import log
import io, re

# ========================
# Constants / maps
# ========================
DIGITS = list(range(10))
EVEN = {0,2,4,6,8}
ODD  = {1,3,5,7,9}
LOW  = {0,1,2,3,4}
HIGH = {5,6,7,8,9}
V_TRAC_GROUPS = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}
MIRROR_PAIRS  = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
MIRROR = MIRROR_PAIRS
V_TRAC = V_TRAC_GROUPS
VTRAC_GROUPS = V_TRAC_GROUPS
vtrac = V_TRAC_GROUPS
mirror = MIRROR

# ========================
# Helpers
# ========================
def to_digits(s):
    s = str(s).strip()
    ds = [int(c) for c in s if c.isdigit()]
    if len(ds) > 5: ds = ds[-5:]
    if len(ds) < 5: ds = ([0]*(5-len(ds))) + ds
    return ds

def eo_counts(digs):
    e = sum(1 for d in digs if d in EVEN)
    o = 5 - e
    return f"{e}E/{o}O"

def hl_counts(digs):
    h = sum(1 for d in digs if d in HIGH)
    l = 5 - h
    return f"{h}H/{l}L"

def sum_category_both(total: int):
    if 0 <= total <= 15:
        return "Very Low", "VeryLow"
    elif 16 <= total <= 24:
        return "Low", "Low"
    elif 25 <= total <= 33:
        return "Mid", "Mid"
    else:
        return "High", "High"

def structure_of(digits):
    cnts = sorted(Counter(digits).values(), reverse=True)
    if cnts == [1,1,1,1,1]: return 'SINGLE'
    if cnts == [2,1,1,1]:   return 'DOUBLE'
    if cnts == [2,2,1]:     return 'DOUBLE-DOUBLE'
    if cnts == [3,1,1]:     return 'TRIPLE'
    if cnts == [3,2]:       return 'TRIPLE-DOUBLE'
    if cnts == [4,1]:       return 'QUAD'
    if cnts == [5]:         return 'QUINT'
    return f'OTHER-{cnts}'

def sort_box_str(digs):
    return ''.join(map(str, sorted(digs)))

def wilson_lower_bound(k, n, z=1.96):
    if n <= 0: return 0.0
    phat = k/n
    denom = 1 + z*z/n
    centre = phat + (z*z)/(2*n)
    adj = z * ((phat*(1-phat) + z*z/(4*n))/n) ** 0.5
    return (centre - adj)/denom

def safe_eval(expr, local_vars):
    allowed = {"sum":sum,"any":any,"all":all,"len":len,"min":min,"max":max,"sorted":sorted,
               "set":set,"list":list,"tuple":tuple,"abs":abs,"range":range,"Counter":Counter}
    return eval(expr, {"__builtins__": {}}, {**allowed, **local_vars})

# ========================
# Parse history (robust TXT/CSV)
# ========================
def parse_history_upload(file):
    name = file.name.lower()
    content = file.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        text = content.decode("utf-8", errors="ignore")
        dates, results = [], []
        for line in text.strip().splitlines():
            parts = re.split(r"\t+|\s{2,}", line.strip())
            parts = [p for p in parts if p]
            if len(parts) >= 2 and re.fullmatch(r"\d{5}", parts[-1]):
                dates.append(parts[0]); results.append(parts[-1]); continue
            m = re.search(r"(\d{5})(?!.*\d)", line)
            if m:
                dates.append(line[:m.start()].strip())
                results.append(m.group(1))
        df = pd.DataFrame({"Date": dates, "Result": results})
    if "Result" not in df.columns:
        raise ValueError("History file must contain a 'Result' column or parsable 5-digit tokens.")
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            try:
                df["Date"] = pd.to_datetime(df["Date"].astype(str).str.replace(r"^[A-Za-z]{3},\s*", "", regex=True))
            except Exception:
                pass
    return df

# ========================
# Follower / Carry
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
    if not cnt: return []
    return [d for d,_ in cnt.most_common(top_n)]

def carry_rate_map(train_pairs):
    seen = Counter(); carried = Counter()
    for seed, win in train_pairs:
        for d in set(seed):
            seen[d] += 1
            if d in win: carried[d] += 1
    return {d: (carried[d]/seen[d] if seen[d] else 0.0) for d in DIGITS}

def top2_carry_candidates(prev_digits, carry_rates):
    if not prev_digits: return []
    digs = sorted(set(prev_digits))
    digs.sort(key=lambda d: (-carry_rates.get(d,0.0), d))
    return digs[:2]

# ========================
# Seed feature pack
# ========================
def features_dict(seed_digits, prev_digits, prev2_digits):
    total = sum(seed_digits)
    sc_text, sc_compact = sum_category_both(total)
    spread = (max(seed_digits)-min(seed_digits)) if seed_digits else 0
    if spread <= 3: band = "0-3"
    elif spread <= 6: band = "4-6"
    elif spread <= 9: band = "7-9"
    else: band = "10+"
    return {
        "sum_cat": sc_text,
        "sum_cat_compact": sc_compact,
        "eo": eo_counts(seed_digits),
        "hl": hl_counts(seed_digits),
        "spread_band": band,
        "carry_count": len(set(seed_digits) & set(prev_digits)),
        "mirror_hit": int(any((MIRROR[d] in prev_digits) for d in set(seed_digits))),
        "lead_change": int(len(prev_digits)==5 and seed_digits[0] != prev_digits[0]),
        "trail_change": int(len(prev_digits)==5 and seed_digits[-1] != prev_digits[-1]),
    }

# ========================
# Step-0 generator
# ========================
def step0_generate_pool(seed_digits):
    c = Counter(seed_digits)
    pairs = set()
    for d,n in c.items():
        if n >= 2:
            pairs.add(tuple(sorted([d,d])))
    uniq = sorted(c.keys())
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            pairs.add(tuple(sorted([uniq[i],uniq[j]])))
    if not pairs: return []
    boxes = set()
    for x in range(10):
        for y in range(x,10):
            for z in range(y,10):
                add3 = [x,y,z]
                for p in pairs:
                    combo = list(p) + add3
                    cnts = Counter(combo)
                    if all(v <= 2 for v in cnts.values()):
                        boxes.add(sort_box_str(combo))
    return sorted(boxes)

# ========================
# Build expression context
# ========================
def build_context_vars(seed_row, follower_mat, carry_rates, combo_digits):
    seed = seed_row["seed_digits"]
    prev = seed_row["prev_digits"]
    prev2 = seed_row["prev2_digits"]
    prev3 = seed_row.get("prev3_digits", [])

    follower_top6 = follower_pool_for_seed(seed, follower_mat, 6)
    follower_top2 = follower_pool_for_seed(seed, follower_mat, 2)

    c2 = top2_carry_candidates(prev, carry_rates)
    seedp1_set = sorted({(d+1) % 10 for d in seed})
    union2 = sorted(set(c2) | set(seedp1_set))

    seed_pos = seed[:] if len(seed)==5 else (seed+[None]*5)[:5]
    p1_pos   = [ (d+1) % 10 for d in seed_pos ]

    combo_sum = sum(combo_digits)
    seed_sum  = sum(seed); prev_sum = sum(prev); prev2_sum = sum(prev2); prev3_sum = sum(prev3)

    hot6_win  = seed_row["hot_last20"]
    hot7_win  = seed_row.get("hot7_last10", []) or seed_row.get("hot7_last20", [])
    cold_win  = seed_row["cold_last20"]
    due_last2 = seed_row["due_last2"]

    common_to_both = set(seed) & set(prev)
    last2_union    = sorted(set(seed) | set(prev))

    seed_vtracs  = set(V_TRAC_GROUPS[d] for d in seed)
    combo_vtracs = set(V_TRAC_GROUPS[d] for d in combo_digits)

    sc_text, sc_compact = sum_category_both(seed_sum)

    loc = {
        "DIGITS": DIGITS, "EVEN": EVEN, "ODD": ODD, "LOW": LOW, "HIGH": HIGH, "MIRROR": MIRROR,
        "seed_digits": seed, "prev_digits": prev, "prev2_digits": prev2,

        "seed_value": int(''.join(map(str, seed))) if len(seed)==5 else None,
        "seed_sum": seed_sum, "prev_sum": prev_sum, "prev_seed_sum": prev_sum,
        "prev_prev_seed_sum": prev2_sum, "prev_prev_prev_seed_sum": prev3_sum,

        "seed_digits_1": prev, "seed_digits_2": prev2, "seed_digits_3": prev3,
        "prev_seed_digits": prev, "prev_prev_seed_digits": prev2, "prev_prev_prev_seed_digits": prev3,

        "new_seed_digits": set(seed) - set(prev),
        "prev_pattern": (sc_text, ("Even" if seed_sum %2==0 else "Odd")),
        "hot_digits": hot6_win, "cold_digits": cold_win, "due_digits": due_last2,

        "seed_counts": Counter(seed),
        "combo_digits": combo_digits, "combo_sum": combo_sum,
        "combo_sum_cat": sc_text,
        "combo_structure": structure_of(combo_digits),
        "winner_structure": structure_of(seed_row["winner_digits"]),

        "seed_vtracs": seed_vtracs, "combo_vtracs": combo_vtracs,

        "common_to_both": common_to_both, "last2": set(last2_union),

        "follower_top6": follower_top6, "follower_top2": follower_top2,
        "carry2": c2, "c1": c2,
        "u1": union2, "u2": union2, "u3": union2, "u4": union2, "u5": union2, "u6": union2, "u7": union2,

        "seedp1": seedp1_set,
        "seed_plus_1": seedp1_set,
        "s1": [seed_pos[0]], "s2": [seed_pos[1]], "s3": [seed_pos[2]], "s4": [seed_pos[3]], "s5": [seed_pos[4]],
        "p1": seedp1_set, "p2": [p1_pos[1]], "p3": [p1_pos[2]], "p4": [p1_pos[3]], "p5": [p1_pos[4]],

        "UNION_DIGITS": sorted(set(seed) | set(seedp1_set)),
        "union_digits": sorted(set(seed) | set(seedp1_set)),

        "hot_last20": hot6_win, "hot7_last20": seed_row.get("hot7_last20", []), "hot7_last10": hot7_win,
        "cold_last20": cold_win,
        "due_last2": due_last2,

        "mirror": MIRROR, "MIRROR_PAIRS": MIRROR_PAIRS,
        "mirror_of": lambda d: MIRROR.get(d),

        "digit_prev_letters": {}, "digit_current_letters": {},
        "prev_core_letters": set(), "core_letters_prevmap": [],

        "is_even": lambda d: d in EVEN, "is_odd": lambda d: d in ODD,
        "is_low": lambda d: d in LOW, "is_high": lambda d: d in HIGH,
        "V_TRAC_GROUPS": V_TRAC_GROUPS, "V_TRAC": V_TRAC_GROUPS, "VTRAC_GROUPS": V_TRAC_GROUPS,
        "vtrac": V_TRAC_GROUPS,

        "nan": float("nan"),
    }
    return loc

def evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, combo_digits):
    try:
        loc = build_context_vars(seed_row, follower_mat, carry_rates, combo_digits)
        if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
            expr = eval(expr)
        out = safe_eval(expr, loc)
        return not bool(out)
    except Exception:
        return True

def winner_kept_by_filter(expr, seed_row, follower_mat, carry_rates):
    return evaluate_filter_on_combo(expr, seed_row, follower_mat, carry_rates, seed_row["winner_digits"])

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="DC-5 Seed-Aware Filter Recommender v5 (fix)", layout="wide")
st.title("DC-5 Seed-Aware Filter Recommender — Tester-Compatible (v5, diagnostics-safe)")

with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Draw history (CSV or TXT)", type=["csv","txt"])
    filt_file = st.file_uploader("Filter CSV (5-col id,name,enabled,applicable_if,expression)", type=["csv"])
    hot_win = st.number_input("Hot/Cold window", 5, 100, 20, 1)
    flip_reverse = st.checkbox("Flip if file is reverse-chron (newest→oldest)", value=True)
    min_bucket_n = st.number_input("Min bucket sample size", 5, 200, 20, 1)
    keep_lb_thresh = st.slider("Min Wilson-LB KeepRate (SAFE)", 0.50, 1.00, 0.90, 0.01)
    elim_min = st.slider("Min elimination fraction", 0.0, 0.8, 0.20, 0.01)
    current_idx = st.number_input("Target seed index (0 = earliest after ordering)", 0, 9999, 0, 1)
    run_btn = st.button("Run")

def normalize_filter_df(df):
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if "id" not in df.columns and "ID" in df.columns: df["id"] = df["ID"]
    if "name" not in df.columns and "Name" in df.columns: df["name"] = df["Name"]
    if "enabled" not in df.columns:
        df["enabled"] = df["Enabled"] if "Enabled" in df.columns else True
    if "applicable_if" not in df.columns:
        df["applicable_if"] = df["applicable_if"] if "applicable_if" in df.columns else ""
    if "expression" not in df.columns and "Expression" in df.columns: df["expression"] = df["Expression"]
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    def pbool(x):
        if isinstance(x,bool): return x
        s = str(x).strip().lower()
        return s not in {"false","0","no","off",""}
    df["enabled"] = df["enabled"].map(pbool)
    df["applicable_if"] = df["applicable_if"].fillna("").astype(str)
    df["expression"] = df["expression"].fillna("").astype(str)
    return df[["id","name","enabled","applicable_if","expression"]]

if run_btn:
    if not hist_file or not filt_file:
        st.error("Please provide both files.")
        st.stop()

    # Parse history
    try:
        hist_df = parse_history_upload(hist_file)
    except Exception as e:
        st.error(f"History parse error: {e}")
        st.stop()

    # Order rows earliest→latest
    if "Date" in hist_df.columns and pd.api.types.is_datetime64_any_dtype(hist_df["Date"]):
        asc = hist_df["Date"].is_monotonic_increasing
        desc = hist_df["Date"].is_monotonic_decreasing
        if desc and flip_reverse:
            hist_df = hist_df.sort_values("Date", ascending=True).reset_index(drop=True)
        elif not asc:
            hist_df = hist_df.sort_values("Date", ascending=True).reset_index(drop=True)
    else:
        if flip_reverse:
            hist_df = hist_df.iloc[::-1].reset_index(drop=True)

    st.subheader("History preview")
    st.write(f"Rows found: **{len(hist_df)}**")
    st.dataframe(hist_df.head(5), use_container_width=True)
    st.dataframe(hist_df.tail(5), use_container_width=True)

    if "Result" not in hist_df.columns or len(hist_df) < 6:
        st.error("Need ≥6 rows with 'Result'.")
        st.stop()

    # Build rows (walk-forward pairs)
    results = hist_df["Result"].astype(str).tolist()
    rows = []
    for i in range(len(results)-1):
        seed = to_digits(results[i]); win = to_digits(results[i+1])
        prev = to_digits(results[i-1]) if i-1 >= 0 else []
        prev2 = to_digits(results[i-2]) if i-2 >= 0 else []
        prev3 = to_digits(results[i-3]) if i-3 >= 0 else []

        # rolling window (hot/cold) + last-10 hot7
        start = max(0, i-int(hot_win))
        window_vals = [d for s in results[start:i] for d in to_digits(s)]
        cnt = Counter(window_vals)
        hot_sorted = [d for d,_ in cnt.most_common()]
        cold_sorted = [d for d,_ in cnt.most_common()[::-1]]
        hot6 = hot_sorted[:6] if hot_sorted else []
        hot7_20 = hot_sorted[:7] if hot_sorted else []

        start10 = max(0, i-10)
        win10_vals = [d for s in results[start10:i] for d in to_digits(s)]
        cnt10 = Counter(win10_vals)
        hot7_10 = [d for d,_ in cnt10.most_common(7)] if cnt10 else []

        due2 = sorted(set(range(10)) - set(prev) - set(prev2))
        s2 = sorted({(d+1)%10 for d in seed})
        union_digits = sorted(set(seed) | set(s2))

        rows.append({
            "i": i,
            "seed_digits": seed,
            "winner_digits": win,
            "prev_digits": prev,
            "prev2_digits": prev2,
            "prev3_digits": prev3,
            "hot_last20": hot6,
            "hot7_last20": hot7_20,
            "hot7_last10": hot7_10,
            "cold_last20": cold_sorted[:6] if cold_sorted else [],
            "due_last2": due2,
            "s2": s2,
            "UNION_DIGITS": union_digits,
        })

    # filter CSV
    try:
        filt_df_raw = pd.read_csv(filt_file)
    except Exception as e:
        st.error(f"Filter CSV read error: {e}")
        st.stop()
    filt_df = normalize_filter_df(filt_df_raw)
    filt_df = filt_df[(filt_df["expression"].str.len()>0) & (filt_df["enabled"]==True)]

    # Precompute prior-only keeps per filter
    def follower_upto(idx):
        return follower_matrix([{"seed_digits": rr["seed_digits"], "winner_digits": rr["winner_digits"]} for rr in rows[:idx]])
    def carryrates_upto(idx):
        return carry_rate_map([(rr["seed_digits"], rr["winner_digits"]) for rr in rows[:idx]])

    per_index_keep = defaultdict(dict)
    for i, r in enumerate(rows):
        if i>0:
            fm = follower_upto(i); cr = carryrates_upto(i)
        else:
            fm = {d: Counter() for d in DIGITS}; cr = {d:0.0 for d in DIGITS}
        for _, f in filt_df.iterrows():
            if not f["enabled"]: continue
            app = (f["applicable_if"] or "").strip()
            applicable = True
            if app:
                try:
                    if (app.startswith('"') and app.endswith('"')) or (app.startswith("'") and app.endswith("'")):
                        app = eval(app)
                    applicable = bool(safe_eval(app, build_context_vars(r, fm, cr, r["seed_digits"])))
                except Exception:
                    applicable = True
            if not applicable:
                per_index_keep[i][f["id"]] = None
                continue
            per_index_keep[i][f["id"]] = int(winner_kept_by_filter(f["expression"], r, fm, cr))

    cur = int(current_idx)
    if cur < 0 or cur >= len(rows):
        st.error(f"Target seed index out of range. Valid 0..{len(rows)-1}.")
        st.stop()

    rcur = rows[cur]

    # prior bucket stats
    def bkey(row):
        f = features_dict(row["seed_digits"], row["prev_digits"], row["prev2_digits"])
        return tuple(sorted(f.items()))

    stats = defaultdict(lambda: {"k":0,"n":0})
    for i in range(cur):
        key = bkey(rows[i])
        for _, f in filt_df.iterrows():
            val = per_index_keep[i].get(f["id"])
            if val is None: continue
            stats[(f["id"], key)]["n"] += 1
            stats[(f["id"], key)]["k"] += int(val==1)

    # step-0 pool for current seed
    pool = step0_generate_pool(rcur["seed_digits"])

    # now score each filter at current bucket
    if cur>0:
        fm_cur = follower_upto(cur); cr_cur = carryrates_upto(cur)
    else:
        fm_cur = {d: Counter() for d in DIGITS}; cr_cur = {d:0.0 for d in DIGITS}

    recs = []
    key_cur = bkey(rcur)
    for _, f in filt_df.iterrows():
        if not f["enabled"]: continue
        app = (f["applicable_if"] or "").strip()
        applicable = True
        if app:
            try:
                if (app.startswith('"') and app.endswith('"')) or (app.startswith("'") and app.endswith("'")):
                    app = eval(app)
                applicable = bool(safe_eval(app, build_context_vars(rcur, fm_cur, cr_cur, rcur["seed_digits"])))
            except Exception:
                applicable = True
        if not applicable: continue

        kept = stats[(f["id"], key_cur)]["k"]
        total = stats[(f["id"], key_cur)]["n"]
        lb = wilson_lower_bound(kept, total)

        survive = 0
        for box in pool:
            digs = [int(c) for c in box]
            if evaluate_filter_on_combo(f["expression"], rcur, fm_cur, cr_cur, digs):
                survive += 1
        elim = (len(pool)-survive)/len(pool) if pool else 0.0

        tag = "Safe" if (lb >= keep_lb_thresh and elim >= elim_min and total >= min_bucket_n) else ("Watch" if lb >= 0.80 else "Risky")
        role = "LoserList" if str(f["id"]).strip().upper().startswith("LL") else ""

        recs.append({
            "filter_id": f["id"],
            "name": f["name"],
            "Role": role,
            "keep_kept": kept,
            "keep_total": total,
            "keep_LB": round(lb,3),
            "elim_frac_current": round(elim,3),
            "elim_count_current": int(round(elim*len(pool))) if pool else 0,
            "pool_size": len(pool),
            "Score": round(lb * log(1 + max(0.0, elim)),6),
            "Tag": tag,
        })

    rec_df = pd.DataFrame(recs).sort_values(["Tag","Score","keep_LB"], ascending=[True, False, False]).reset_index(drop=True)

    st.subheader("Seed context (current)")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write("Seed:", rcur["seed_digits"])
        st.write("Prev:", rcur["prev_digits"])
        st.write("Prev2:", rcur["prev2_digits"])
        st.write("Due last2:", rcur["due_last2"])
    with c2:
        sc_text,_ = sum_category_both(sum(rcur["seed_digits"]))
        st.write("Sum cat:", sc_text)
        st.write("EO:", eo_counts(rcur["seed_digits"]))
        st.write("HL:", hl_counts(rcur["seed_digits"]))
    with c3:
        st.write("Step-0 size:", len(pool))

    st.subheader("Recommendations (prior-only stats)")
    if len(rec_df)==0:
        st.info("No applicable filters produced scores at this index with current thresholds.")
    else:
        st.dataframe(rec_df, use_container_width=True)

    # Diagnostics: feature → safety
    st.markdown("### Diagnostics: Seed characteristics → Filter safety")
    if cur == 0:
        st.info("Diagnostics need prior rows. Increase the Target seed index and rerun.")
        st.stop()
    if len(rec_df)==0:
        st.info("No applicable filters to analyze at this index.")
        st.stop()

    choices = [f"{r['filter_id']} — {r['name']}" for _, r in rec_df.iterrows()]
    pick = st.selectbox("Pick filter", choices, index=0)
    pick_id = pick.split(" — ")[0]

    # Build feature frame for prior rows
    feat_rows = []
    for i, r in enumerate(rows):
        f = features_dict(r["seed_digits"], r["prev_digits"], r["prev2_digits"])
        f["i"] = i
        feat_rows.append(f)
    feat_df = pd.DataFrame(feat_rows)

    feat_cols = ["sum_cat","sum_cat_compact","eo","hl","spread_band","carry_count","mirror_hit","lead_change","trail_change"]
    prior = list(range(cur))
    rows_out = []
    for feat in feat_cols:
        for i in prior:
            val = feat_df.loc[i, feat]
            keep_val = per_index_keep[i].get(pick_id)
            if keep_val is None: continue
            rows_out.append({"feature": feat, "value": val,
                             "kept": int(keep_val==1), "total": 1,
                             "matches_current": bool(val == feat_df.loc[cur, feat])})
    if not rows_out:
        st.info("No prior rows where this filter was applicable. Try a later target index or another filter.")
        st.stop()

    # aggregate
    diag_df = pd.DataFrame(rows_out)
    diag_df = diag_df.groupby(["feature","value","matches_current"], as_index=False).agg({"kept":"sum","total":"sum"})
    diag_df["keep_LB"] = diag_df.apply(lambda r: wilson_lower_bound(r["kept"], r["total"]), axis=1)
    sort_cols = [c for c in ["matches_current","keep_LB","total"] if c in diag_df.columns]
    diag_df = diag_df.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)

    st.dataframe(diag_df, use_container_width=True)

    # Download recs
    csv_buf = io.StringIO()
    rec_df.to_csv(csv_buf, index=False)
    st.download_button("Download recommendations CSV", data=csv_buf.getvalue(), file_name="filter_recommendations_v5_fix.csv", mime="text/csv")
