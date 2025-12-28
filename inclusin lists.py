# dc5_inclusion_lists_app.py
# DC-5 Inclusion Lists (≥70% Wilson-LB) — Seed-Conditioned Hot/Cold/Due
# (CSV/TXT friendly + bigger non-scrolling Last-10 input + robust parsing)

import math, re, io
from typing import List, Tuple, Dict, Optional, Tuple as Tup

import pandas as pd
import streamlit as st
from collections import Counter

# ------------------------- Settings / Filenames -------------------------
CALIB_OVERALL = "dc5_inclusion_calib_overall.csv"
CALIB_CONTEXT = "dc5_inclusion_calib_context.csv"
INCL_FAMILIES = [
    "L3_HHH","L3_HHD","L3_HCD","L3_HHC",
    "L4_HHHH","L4_HHCD","L4_HHDD",
]

# ------------------------- Utilities -------------------------

def to_digits(s: str) -> List[int]:
    s = str(s)
    digs = [int(c) for c in s if c.isdigit()]
    if len(digs) != 5:
        digs = ([0] * (5 - len(digs))) + digs[-5:]
    return digs

def _extract_5s_from_any_text(text: str) -> List[str]:
    """Return all 5-digit chunks found in arbitrary text (handles spaces, commas, CSV cells)."""
    if not text:
        return []
    # find non-overlapping 5-digit tokens (e.g., '12345')
    tokens = re.findall(r'\b\d{5}\b', text)
    return tokens

def _extract_5s_from_dataframe(df: pd.DataFrame) -> List[str]:
    """Flatten a DataFrame and extract every 5-digit token from any column."""
    chunks: List[str] = []
    for col in df.columns:
        series = df[col].astype(str)
        for cell in series:
            chunks.extend(_extract_5s_from_any_text(cell))
    return chunks

def parse_draws_from_text(text: str) -> List[str]:
    # Wrapper kept for backward-compatibility
    return _extract_5s_from_any_text(text)

def parse_draws_from_upload(file) -> List[str]:
    """Accept CSV/TXT. If CSV, parse with pandas then scan every column; if TXT, scan the text."""
    if file is None:
        return []
    data = file.read()
    # try text decodes
    decoded = None
    for enc in ("utf-8","utf-16","latin-1"):
        try:
            decoded = data.decode(enc)
            break
        except Exception:
            continue
    if decoded is None:
        return []
    # If it looks like CSV (has commas or multiple lines), try pandas first
    looks_like_csv = ("," in decoded) or ("\n" in decoded and ";" in decoded)
    try:
        df = pd.read_csv(io.StringIO(decoded))
        return _extract_5s_from_dataframe(df)
    except Exception:
        # Not a CSV (or failed) → plain text
        return _extract_5s_from_any_text(decoded)

def seed_structure(digits: List[int]) -> str:
    cnts = sorted(Counter(digits).values(), reverse=True)
    if cnts == [1,1,1,1,1]: return "SINGLE"
    if cnts == [2,1,1,1]:   return "DOUBLE"
    if cnts == [2,2,1]:     return "DOUBLE-DOUBLE"
    if cnts == [3,1,1]:     return "TRIPLE"
    if cnts == [3,2]:       return "TRIPLE-DOUBLE"
    if cnts == [4,1]:       return "QUAD"
    if cnts == [5]:         return "QUINT"
    return f"OTHER-{cnts}"

def sum_category(total: int) -> str:
    if 0 <= total <= 15: return "Very Low"
    if 16 <= total <= 24: return "Low"
    if 25 <= total <= 33: return "Mid"
    return "High"

def wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    p = k / n
    denom = 1 + z*z/n
    center = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    lb = (center - adj) / denom
    return max(0.0, lb)

# ------------------------- Heat helpers -------------------------

def ranked_counts(window_results: List[str]) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], Dict[int,int]]:
    counts = Counter(int(c) for r in window_results for c in r if c.isdigit())
    for d in range(10):
        counts.setdefault(d, 0)
    hot_rank = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    cold_rank= sorted(counts.items(), key=lambda x: (x[1], x[0]))
    return hot_rank, cold_rank, counts

def due_set(prev: List[int], prev2: List[int]) -> set:
    return set(range(10)) - set(prev) - set(prev2)

def pick_top_unique(ranked_list: List[Tuple[int,int]], k: int, banned: set = set()) -> List[int]:
    out, seen = [], set(banned)
    for d, _ in ranked_list:
        if d in seen: continue
        out.append(d); seen.add(d)
        if len(out) == k: break
    return out

def pick_best_due(due: set, counts: Dict[int,int], banned: set = set()) -> Optional[int]:
    lst = sorted([(d, counts.get(d,0)) for d in due if d not in banned], key=lambda x: (-x[1], x[0]))
    return lst[0][0] if lst else None

# ------------------------- Inclusion lists for a seed -------------------------

def current_seed_lists(last10: List[str], heat_window: int) -> Dict[str, List[int]]:
    if len(last10) < heat_window:
        raise ValueError(f"Need at least {heat_window} draws; last is the current seed.")
    window = last10[-heat_window:-1]
    prev = to_digits(last10[-2]) if len(last10) >= 2 else []
    prev2= to_digits(last10[-3]) if len(last10) >= 3 else []
    hot_rank, cold_rank, counts = ranked_counts(window)
    _due = due_set(prev, prev2)

    lists: Dict[str, List[int]] = {}
    lists["L3_HHH"] = pick_top_unique(hot_rank, 3)

    t2 = pick_top_unique(hot_rank, 2)
    bd = pick_best_due(_due, counts, banned=set(t2))
    lists["L3_HHD"] = t2 + ([bd] if bd is not None else pick_top_unique(hot_rank, 1, banned=set(t2)))

    l = pick_top_unique(hot_rank, 1)
    l += pick_top_unique(cold_rank, 1, banned=set(l))
    bd2 = pick_best_due(_due, counts, banned=set(l))
    if bd2 is not None: l += [bd2]
    while len(l) < 3:
        l += pick_top_unique(hot_rank, 1, banned=set(l))
    lists["L3_HCD"] = l[:3]

    l = pick_top_unique(hot_rank, 2)
    l += pick_top_unique(cold_rank, 1, banned=set(l))
    while len(l) < 3:
        l += pick_top_unique(hot_rank, 1, banned=set(l))
    lists["L3_HHC"] = l[:3]

    lists["L4_HHHH"] = pick_top_unique(hot_rank, 4)

    l = pick_top_unique(hot_rank, 2)
    l += pick_top_unique(cold_rank, 1, banned=set(l))
    bd3 = pick_best_due(_due, counts, banned=set(l))
    if bd3 is not None: l += [bd3]
    while len(l) < 4:
        l += pick_top_unique(hot_rank, 1, banned=set(l))
    lists["L4_HHCD"] = l[:4]

    l = pick_top_unique(hot_rank, 2)
    due_sorted = sorted([(d, counts.get(d,0)) for d in _due if d not in l], key=lambda x: (-x[1], x[0]))
    for d, _ in due_sorted[:2]:
        l.append(d)
    while len(l) < 4:
        l += pick_top_unique(hot_rank, 1, banned=set(l))
    lists["L4_HHDD"] = l[:4]

    return lists

# ------------------------- Calibration from history -------------------------

def build_transitions(history_results: List[str], heat_window: int) -> pd.DataFrame:
    results = history_results[:]
    rows: List[Dict] = []
    for i in range(len(results) - 1):
        start = max(0, i - heat_window)
        window = results[start:i]
        if len(window) == 0:
            continue
        seed_s = results[i]
        win_s  = results[i+1]
        seed = to_digits(seed_s)
        win  = to_digits(win_s)
        # mirror current heat computation
        lists = current_seed_lists(results[max(0, i - heat_window): i+1], heat_window)

        def hit(key: str) -> int:
            S = set(lists[key])
            return int(any(d in S for d in win))

        rows.append({
            "Seed": seed_s,
            "Winner": win_s,
            "SeedStruct": seed_structure(seed),
            "SeedSumCat": sum_category(sum(seed)),
            **{f"Hit_{k}": hit(k) for k in INCL_FAMILIES}
        })
    return pd.DataFrame(rows)

def summarize_calibration(df: pd.DataFrame) -> Tup[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in df.columns if c.startswith("Hit_")]
    n = len(df)
    ov_rows = []
    for col in cols:
        k = int(df[col].sum())
        p = k/n if n else 0.0
        lb = wilson_lb(k, n)
        ov_rows.append({"List": col.replace("Hit_",""), "Scope": "OVERALL", "n": n, "Hits": k,
                        "P %": round(100*p,2), "Wilson LB %": round(100*lb,2)})
    overall = pd.DataFrame(ov_rows).sort_values("P %", ascending=False)

    ctx_rows = []
    for (s,c), g in df.groupby(["SeedStruct","SeedSumCat"], dropna=False):
        nn = len(g)
        for col in cols:
            kk = int(g[col].sum())
            pp = kk/nn if nn else 0.0
            lb = wilson_lb(kk, nn)
            ctx_rows.append({
                "List": col.replace("Hit_",""), "Scope": f"{s} × {c}", "SeedStruct": s, "SeedSumCat": c,
                "n": nn, "Hits": kk, "P %": round(100*pp,2), "Wilson LB %": round(100*lb,2)
            })
    context = pd.DataFrame(ctx_rows).sort_values(["SeedStruct","SeedSumCat","P %"], ascending=[True,True,False])
    return overall, context

# ------------------------- UI -------------------------

st.set_page_config(page_title="DC-5 Inclusion Lists (≥70% LB)", layout="wide")
st.title("DC-5 Inclusion Lists — Seed-Conditioned (≥70% Wilson-LB)")

with st.sidebar:
    st.header("Settings")
    st.session_state.setdefault("heat_window", 10)
    heat_window = st.number_input("Heat window (draws)", min_value=6, max_value=20, value=st.session_state.heat_window, step=1)
    st.session_state.heat_window = int(heat_window)
    lb_thresh = st.slider("Minimum Wilson LB to display (%)", 60, 95, 70, step=1)
    dedup_jaccard = st.slider("De-duplication Jaccard max overlap", 0.0, 1.0, 0.5, step=0.05)
    max_lists_3 = st.number_input("Max 3-digit lists", 1, 10, 4, step=1)
    max_lists_4 = st.number_input("Max 4-digit lists", 1, 10, 4, step=1)

colL, colR = st.columns([1,1])

with colL:
    st.subheader("Last 10 draws (oldest → newest; last = current seed)")
    # Larger height to avoid scrolling; plus a CSV/TXT uploader for convenience
    last10_text = st.text_area("Paste 10 five-digit draws (spaces/commas/lines OK)", height=220, help="Example: 30209 26418 22305 74762 67539 59188 48956 03018 31185 79477")
    last10_file = st.file_uploader("…or upload a CSV/TXT containing your last 10", type=["csv","txt"], key="last10_upl")
    order_flip_last10 = st.checkbox("My last-10 is most-recent-first (reverse)", value=False)

    # live preview of parsed draws
    parsed_last10: List[str] = []
    if last10_file is not None:
        parsed_last10 = parse_draws_from_upload(last10_file)
    elif last10_text.strip():
        parsed_last10 = parse_draws_from_text(last10_text)

    if order_flip_last10 and parsed_last10:
        parsed_last10 = parsed_last10[::-1]

    st.caption(f"Parsed last-10 count: **{len(parsed_last10)}** → {parsed_last10[-10:] if parsed_last10 else []}")

with colR:
    st.subheader("Calibration mode")
    mode = st.radio("How should probabilities be obtained?", (
        "Use cached calibration (fast)",
        "Recalculate from recent history (fresh)",
    ))

    recent_limit = None
    save_cache = False
    hist_results: List[str] = []

    if mode == "Recalculate from recent history (fresh)":
        hist_file = st.file_uploader("Upload history (TXT/CSV)", type=["txt","csv"], key="hist_upl")
        hist_text = st.text_area("…or paste a long history block", height=180)
        hist_order_flip = st.checkbox("My history is most-recent-first (reverse)", value=True)
        recent_limit = st.number_input("Limit to most recent N draws (0 = use all)", min_value=0, max_value=100000, value=0, step=50)
        save_cache = st.checkbox("After recalculation, save tables as default cache", value=True)

        if hist_file is not None:
            hist_results = parse_draws_from_upload(hist_file)
        elif hist_text.strip():
            hist_results = parse_draws_from_text(hist_text)

        if hist_order_flip:
            hist_results = hist_results[::-1]
        if recent_limit and recent_limit > 0:
            hist_results = hist_results[-int(recent_limit):]

        st.caption(f"Parsed history count: **{len(hist_results)}** (oldest → newest preview: {hist_results[:5]} … {hist_results[-5:]})")

run = st.button("Compute inclusion lists (≥LB threshold)")

if run:
    # Use the ten most recent from whatever was parsed for last10 input
    last10 = parsed_last10[-int(heat_window):] if parsed_last10 else []
    if len(last10) != int(heat_window):
        st.error(f"Please supply exactly {int(heat_window)} draws (found {len(last10)}). The last one must be the current seed.")
        st.stop()

    try:
        lists_digits = current_seed_lists(last10, int(heat_window))
    except Exception as e:
        st.error(f"Failed to build lists: {e}")
        st.stop()

    if mode == "Use cached calibration (fast)":
        try:
            overall = pd.read_csv(CALIB_OVERALL)
            context = pd.read_csv(CALIB_CONTEXT)
        except Exception:
            st.error("No cached calibration found. Switch to 'Recalculate from recent history' once to create it.")
            st.stop()
    else:
        if len(hist_results) < 2:
            st.error("Need at least 2 draws in history to recalculate.")
            st.stop()
        try:
            df_trans = build_transitions(hist_results, int(heat_window))
            overall, context = summarize_calibration(df_trans)
            if save_cache:
                overall.to_csv(CALIB_OVERALL, index=False)
                context.to_csv(CALIB_CONTEXT, index=False)
        except Exception as e:
            st.error(f"Recalculation failed: {e}")
            st.stop()

    seed_digits = to_digits(last10[-1])
    ctx_struct = seed_structure(seed_digits)
    ctx_sumcat = sum_category(sum(seed_digits))

    out_rows = []
    for fam in INCL_FAMILIES:
        ro = overall[overall["List"] == fam]
        row_o = ro.iloc[0] if len(ro) else None
        P, LB, n, scope = None, None, None, None
        if row_o is not None:
            P, LB, n = row_o["P %"], row_o["Wilson LB %"], int(row_o["n"]) if "n" in row_o else None
            scope = "OVERALL"
        rc = context[(context["List"] == fam) & (context["SeedStruct"] == ctx_struct) & (context["SeedSumCat"] == ctx_sumcat)]
        if len(rc):
            P_c, LB_c = rc.iloc[0]["P %"], rc.iloc[0]["Wilson LB %"]
            n_c = int(rc.iloc[0]["n"]) if "n" in rc.iloc[0] else None
            if LB is None or LB_c >= LB:
                P, LB, n, scope = P_c, LB_c, n_c, f"{ctx_struct} × {ctx_sumcat}"
        digits = lists_digits.get(fam, [])
        out_rows.append({
            "Family": fam,
            "Digits": "{" + ",".join(map(str, digits)) + "}",
            "Scope Used": scope,
            "n": n,
            "P %": P,
            "Wilson LB %": LB,
            "Size": 3 if fam.startswith("L3_") else 4,
        })

    table = pd.DataFrame(out_rows).dropna(subset=["Wilson LB %"]).sort_values(["Size","Wilson LB %","P %"], ascending=[True, False, False])

    t3 = table[(table["Size"]==3) & (table["Wilson LB %"]>=float(lb_thresh))].copy()
    t4 = table[(table["Size"]==4) & (table["Wilson LB %"]>=float(lb_thresh))].copy()

    def jacc(a: str, b: str) -> float:
        A = {int(x) for x in a if x.isdigit()}
        B = {int(x) for x in b if x.isdigit()}
        if not A and not B: return 1.0
        return len(A & B) / len(A | B)

    def dedup(df: pd.DataFrame, cap: int) -> pd.DataFrame:
        chosen = []
        for _, r in df.iterrows():
            if len(chosen) >= cap: break
            if all(jacc(r["Digits"], c["Digits"]) <= float(dedup_jaccard) for c in chosen):
                chosen.append(r)
        return pd.DataFrame(chosen)

    keep_3 = dedup(t3, int(max_lists_3))
    keep_4 = dedup(t4, int(max_lists_4))

    st.subheader("✅ Inclusion lists (≥LB threshold)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**3-digit lists**")
        if keep_3.empty:
            st.info("No 3-digit lists met the LB threshold.")
        else:
            st.dataframe(keep_3[["Family","Digits","Scope Used","n","P %","Wilson LB %"]], use_container_width=True)
    with c2:
        st.markdown("**4-digit lists**")
        if keep_4.empty:
            st.info("No 4-digit lists met the LB threshold.")
        else:
            st.dataframe(keep_4[["Family","Digits","Scope Used","n","P %","Wilson LB %"]], use_container_width=True)

    export_df = pd.concat([keep_3, keep_4], ignore_index=True)
    if not export_df.empty:
        st.download_button("Download lists as CSV", data=export_df.to_csv(index=False).encode("utf-8"), file_name="dc5_inclusion_lists.csv", mime="text/csv")

    with st.expander("Diagnostics — Probability tables in use"):
        st.write(f"Current Seed: **{last10[-1]}** | Structure: **{seed_structure(to_digits(last10[-1]))}** | Sum category: **{sum_category(sum(to_digits(last10[-1])))}**")
        if mode == "Use cached calibration (fast)":
            st.caption("Using cached calibration tables from disk.")
            st.write(f"Loaded: {CALIB_OVERALL}, {CALIB_CONTEXT}")
        else:
            st.caption("Using fresh recalculation from the supplied history. Checked 'save' to persist as cache.")
        st.markdown("**Overall table**")
        st.dataframe(overall, use_container_width=True)
        st.markdown("**Context table**")
        st.dataframe(context, use_container_width=True)

    st.success("Done.")
