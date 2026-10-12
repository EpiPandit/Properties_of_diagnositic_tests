"""
Properties of Diagnostic Tests — Epidemiology 205 (UC Davis)
=============================================================
One Streamlit app for the whole week-3 class activity.

Part 1 (core, done together in class)
  1. Sample animals one at a time and sort them into a 2x2 table on paper
  2. Report the group's 2x2 table + calculations (the app checks the arithmetic)
  3. Live class board comparing all groups (with 95% CIs) -> "why do we disagree?"

Part 2 (going further, once students see that sampling matters)
  4. Does sample size matter?          (batch sampling + repeated studies)
  5. Prevalence & predictive values    (PPV/NPV vs prevalence, apparent vs true prevalence)
  6. Cut-offs & ROC curves             (canine hypothyroidism T4/T3)
  7. Likelihood ratios                 (pre-test -> post-test probability)

Shared class data are stored in a small SQLite file next to this script.
Instructor controls (reveal answers, reset, download CSVs) are password protected;
set INSTRUCTOR_PASSWORD in .streamlit/secrets.toml or the Streamlit Cloud secrets.
"""

import math
import os
import random
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Properties of Diagnostic Tests · Epi 205",
    page_icon="🧪",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
N_GROUPS = 8
DB_PATH = Path(__file__).parent / "class_results.db"

SCENARIOS = {
    "Scenario 1": {"prevalence": 0.10, "sensitivity": 0.90, "specificity": 0.95},
    "Scenario 2": {"prevalence": 0.90, "sensitivity": 0.90, "specificity": 0.95},
    "Scenario 3": {"prevalence": 0.90, "sensitivity": 0.80, "specificity": 0.68},
    "Scenario 4": {"prevalence": 0.20, "sensitivity": 0.85, "specificity": 0.85},
}
BREED_PREVALENCE = {"Golden Retriever": 0.60, "Shih-tzu": 0.05, "Alaskan Malamute": 0.35}
NORMAL_RANGE = {"T4": (12.0, 45.0, "nmol/L"), "T3": (20.0, 90.0, "ng/dL")}

DEFAULT_SETTINGS = {
    "reveal_params": "0",          # show true scenario parameters to students
    "reveal_roc": "0",             # unlock ROC curve / Youden answer in activity 6
    "draws_per_student": "5",      # single-sampling draws allowed per student per scenario
    "active_scenario": "Scenario 1",
}

METRICS = ["Sensitivity", "Specificity", "PPV", "NPV", "True prevalence", "Apparent prevalence"]
FORMULAS = {
    "Sensitivity": "a / (a + c)",
    "Specificity": "d / (b + d)",
    "PPV": "a / (a + b)",
    "NPV": "d / (c + d)",
    "True prevalence": "(a + c) / n",
    "Apparent prevalence": "(a + b) / n",
}
CELL_NAMES = {"a": "true positive", "b": "false positive", "c": "false negative", "d": "true negative"}

# --------------------------------------------------------------------------------------
# Shared storage (SQLite, shared by every student session on the server)
# --------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, grp INTEGER, student TEXT,
    scenario TEXT, diseased INTEGER, test_pos INTEGER);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, grp INTEGER, scenario TEXT,
    a INTEGER, b INTEGER, c INTEGER, d INTEGER,
    se REAL, sp REAL, ppv REAL, npv REAL, true_prev REAL, app_prev REAL, n_correct INTEGER);
CREATE TABLE IF NOT EXISTS thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, grp INTEGER, breed TEXT, test_type TEXT,
    n INTEGER, threshold REAL, tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER, se REAL, sp REAL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


@st.cache_resource
def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", list(DEFAULT_SETTINGS.items())
    )
    conn.commit()
    return conn, threading.Lock()


def db_exec(sql, params=()):
    conn, lock = _db()
    with lock:
        conn.execute(sql, params)
        conn.commit()


def db_query(sql, params=()):
    conn, lock = _db()
    with lock:
        return pd.read_sql_query(sql, conn, params=params)


def get_setting(key):
    df = db_query("SELECT value FROM settings WHERE key = ?", (key,))
    return df["value"].iloc[0] if len(df) else DEFAULT_SETTINGS[key]


def set_setting(key, value):
    db_exec(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def now():
    return datetime.now().isoformat(timespec="seconds")


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def latest_reports(scenario=None):
    """Most recent report from each group (per scenario)."""
    df = db_query(
        "SELECT r.* FROM reports r JOIN "
        "(SELECT grp, scenario, MAX(id) AS mid FROM reports GROUP BY grp, scenario) m "
        "ON r.id = m.mid ORDER BY r.scenario, r.grp"
    )
    if scenario is not None:
        df = df[df["scenario"] == scenario]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Epidemiology helpers
# --------------------------------------------------------------------------------------
def safe_div(x, y):
    return x / y if y else float("nan")


def metric_parts(a, b, c, d):
    """(numerator, denominator) for each measure from a 2x2 table."""
    n = a + b + c + d
    return {
        "Sensitivity": (a, a + c),
        "Specificity": (d, b + d),
        "PPV": (a, a + b),
        "NPV": (d, c + d),
        "True prevalence": (a + c, n),
        "Apparent prevalence": (a + b, n),
    }


def two_by_two_metrics(a, b, c, d):
    return {m: safe_div(x, n) for m, (x, n) in metric_parts(a, b, c, d).items()}


def expected_metrics(p):
    """True (population) values implied by prevalence, sensitivity and specificity."""
    prev, se, sp = p["prevalence"], p["sensitivity"], p["specificity"]
    pos = se * prev + (1 - sp) * (1 - prev)
    return {
        "Sensitivity": se,
        "Specificity": sp,
        "PPV": se * prev / pos,
        "NPV": sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev),
        "True prevalence": prev,
        "Apparent prevalence": pos,
    }


def wilson_ci(x, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{100 * x:.{digits}f}%"


def render_2x2(a=None, b=None, c=None, d=None, fmt="{:,}"):
    """Markdown 2x2 table. With no numbers, shows the symbolic template."""
    if a is None:
        cells = ["**a**<br>true positives", "**b**<br>false positives",
                 "**c**<br>false negatives", "**d**<br>true negatives"]
        tot = ["a + b", "c + d", "a + c", "b + d", "n"]
    else:
        f = fmt.format
        cells = [f(a), f(b), f(c), f(d)]
        tot = [f(a + b), f(c + d), f(a + c), f(b + d), f(a + b + c + d)]
    st.markdown(
        f"""
|  | **Disease +**<br>(gold standard) | **Disease −**<br>(gold standard) | **Total** |
|:--|:--:|:--:|:--:|
| **Test +** | {cells[0]} | {cells[1]} | {tot[0]} |
| **Test −** | {cells[2]} | {cells[3]} | {tot[1]} |
| **Total** | {tot[2]} | {tot[3]} | {tot[4]} |
""",
        unsafe_allow_html=True,
    )


def generate_dog(test_type, breed):
    """Simulated thyroid hormone value and true status for one dog."""
    hypo = random.random() < BREED_PREVALENCE[breed]
    if test_type == "T4":
        value = random.uniform(5, 20) if hypo else random.uniform(15, 50)
        value += random.uniform(-5, 5)
    else:  # T3: more overlap / noise
        value = random.uniform(10, 40) if hypo else random.uniform(30, 100)
        value += random.uniform(-10, 10)
    return round(max(0.0, value), 2), hypo


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def require_group():
    g = st.session_state.get("group")
    if not g:
        st.warning("👈 First choose your **group number** in the sidebar.")
        st.stop()
    return g


def scenario_picker(label="Scenario (your instructor will tell you which one)", key=None):
    names = list(SCENARIOS)
    active = get_setting("active_scenario")
    return st.selectbox(label, names, index=names.index(active) if active in names else 0, key=key)


def params_revealed():
    return get_setting("reveal_params") == "1"


def show_true_params(scenario):
    p = SCENARIOS[scenario]
    st.success(
        f"**{scenario} — true values (revealed by instructor):** "
        f"prevalence {pct(p['prevalence'], 0)}, sensitivity {pct(p['sensitivity'], 0)}, "
        f"specificity {pct(p['specificity'], 0)}"
    )


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def home():
    st.title("🧪 Properties of Diagnostic Tests")
    st.caption("Epidemiology 205 · UC Davis · Interactive class activity")

    st.markdown(
        """
Welcome! In this activity you'll act as a team of veterinarians evaluating a **new screening test**.
You'll sample animals, build your own 2×2 table, and then see how your group's answers compare with
everyone else's. The surprises you hit along the way lead into the rest of the week's ideas.
"""
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🎯 Learning objectives")
        st.markdown(
            """
- **Define** and calculate sensitivity (Se), specificity (Sp), positive predictive value (PPV)
  and negative predictive value (NPV), and describe how they relate to each other.
- **Analyze** test properties together with prior information about disease probability to
  calculate predictive values.
- **Evaluate** whether screening tests are being used judiciously in clinical and public-health settings.
- **Develop** strategies for using diagnostic tests to greater advantage (cut-offs, likelihood ratios).
"""
        )
        st.subheader("🗺️ How the session runs")
        st.markdown(
            """
| Step | What you do | What you'll discover |
|:--|:--|:--|
| **1. Sample animals** | Each person samples and tests 5 animals | How a 2×2 table is built |
| **2. Report your 2×2** | Pool your group's animals on paper, calculate, submit | Whether your arithmetic is right |
| **3. Class results** | Compare all groups' estimates | Same test, different answers. Why? |
| **4. Sample size** | Repeat studies of different sizes | How sampling error shrinks with *n* |
| **5. Prevalence & PVs** | Change prevalence, watch PPV/NPV | Why rare diseases give false alarms |
| **6. Cut-offs & ROC** | Choose a T4 cut-off for hypothyroid dogs | The Se / Sp trade-off |
| **7. Likelihood ratios** | Pre-test → post-test probability | Using a test result on *one* patient |
"""
        )
    with c2:
        st.subheader("📋 Before you start")
        st.markdown(
            """
1. Pick your **group number** in the sidebar (and your initials if you like).
2. Have **paper and a pencil** ready. You'll draw the 2×2 table below yourselves.
3. One person per group will submit the group's table in step 2.
"""
        )
        st.subheader("The 2×2 table you'll fill in")
        render_2x2()
        st.caption("The gold standard tells us the truth; the new test is what we are evaluating.")
        st.info(
            "Work through the pages in the sidebar **in order**. Steps 1–3 are done together in class; "
            "steps 4–7 build on what you find."
        )


def single_sampling():
    group = require_group()
    st.title("🎲 Step 1 · Sample and test animals")
    st.markdown(
        """
Each click samples **one animal** from the population and gives you two results:
its **true status** from a gold-standard test, and the result of the **new screening test**.
Write each animal down and sort it into your 2×2 table on paper.
"""
    )
    scenario = scenario_picker()
    if params_revealed():
        show_true_params(scenario)

    cap = int(get_setting("draws_per_student"))
    my = st.session_state.setdefault(f"draws::{scenario}", [])

    left, right = st.columns([3, 2], gap="large")
    with left:
        if len(my) < cap:
            if st.button("🎲 Sample & test an animal", type="primary", width="stretch"):
                p = SCENARIOS[scenario]
                diseased = random.random() < p["prevalence"]
                positive = random.random() < (p["sensitivity"] if diseased else 1 - p["specificity"])
                my.append({"Animal": len(my) + 1, "diseased": diseased, "positive": positive})
                db_exec(
                    "INSERT INTO draws(ts, grp, student, scenario, diseased, test_pos) VALUES (?,?,?,?,?,?)",
                    (now(), group, st.session_state.get("student") or "", scenario, int(diseased), int(positive)),
                )
        if len(my) < cap:
            st.caption(f"{cap - len(my)} of {cap} animals left for you in {scenario}.")
        else:
            st.success(
                f"You've sampled your {cap} animals. Pool them with your group's animals into **one** "
                "2×2 table on paper, calculate the measures, then go to **Step 2 · Report your 2×2**."
            )

        if my:
            last = my[-1]
            st.markdown(f"#### Animal #{last['Animal']}")
            m1, m2 = st.columns(2)
            m1.metric("True status (gold standard)", "🟥 Diseased" if last["diseased"] else "🟩 Healthy")
            m2.metric("New test result", "➕ Positive" if last["positive"] else "➖ Negative")

            correct = {(True, True): "a", (False, True): "b", (True, False): "c", (False, False): "d"}[
                (last["diseased"], last["positive"])
            ]
            choice = st.radio(
                "Which cell of the 2×2 table does this animal go in?",
                ["a", "b", "c", "d"],
                index=None,
                horizontal=True,
                key=f"cell::{scenario}::{last['Animal']}",
            )
            if choice == correct:
                st.success(f"✅ Correct: cell **{correct}**, a **{CELL_NAMES[correct]}**.")
            elif choice:
                st.error("Not quite. Find the column for its true status, then the row for its test result.")

    with right:
        st.markdown("##### The 2×2 table")
        render_2x2()
        if my:
            st.markdown("##### My animals")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Animal": [d["Animal"] for d in my],
                        "True status": ["Diseased" if d["diseased"] else "Healthy" for d in my],
                        "Test result": ["Positive" if d["positive"] else "Negative" for d in my],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        n_grp = db_query(
            "SELECT COUNT(*) AS n FROM draws WHERE grp = ? AND scenario = ?", (group, scenario)
        )["n"].iloc[0]
        st.metric(f"Animals sampled by Group {group} in {scenario}", int(n_grp))


def group_report():
    group = require_group()
    st.title("📝 Step 2 · Report your group's 2×2 table")
    st.markdown(
        """
Combine **everyone's** animals into a single 2×2 table on paper and calculate the six measures.
Enter them below: the app checks your arithmetic and adds your table to the class board.
*Only one person per group needs to submit; submitting again replaces your group's earlier entry.*
"""
    )
    scenario = scenario_picker()

    with st.form(f"report::{scenario}"):
        st.markdown("**Your group's counts**")
        widths = [1.1, 1, 1]
        h = st.columns(widths)
        h[1].markdown("**Disease +**")
        h[2].markdown("**Disease −**")
        r1 = st.columns(widths)
        r1[0].markdown("**Test +**")
        a = r1[1].number_input("a (true positives)", min_value=0, step=1, key=f"a::{scenario}")
        b = r1[2].number_input("b (false positives)", min_value=0, step=1, key=f"b::{scenario}")
        r2 = st.columns(widths)
        r2[0].markdown("**Test −**")
        c = r2[1].number_input("c (false negatives)", min_value=0, step=1, key=f"c::{scenario}")
        d = r2[2].number_input("d (true negatives)", min_value=0, step=1, key=f"d::{scenario}")

        st.markdown("**Your calculations (as %)**. Leave a box blank if it can't be calculated.")
        cols = st.columns(3)
        answers = {}
        for i, m in enumerate(METRICS):
            answers[m] = cols[i % 3].number_input(
                f"{m} (%)", min_value=0.0, max_value=100.0, value=None, step=0.1,
                format="%.1f", key=f"ans::{scenario}::{m}",
            )
        submitted = st.form_submit_button("✔️ Check & submit to class board", type="primary")

    if submitted:
        a, b, c, d = int(a), int(b), int(c), int(d)
        if a + b + c + d == 0:
            st.error("Your table is empty. Enter your group's counts first.")
            return
        truth = two_by_two_metrics(a, b, c, d)
        rows, n_correct = [], 0
        for m in METRICS:
            t, u = truth[m], answers[m]
            if math.isnan(t):
                status = "➖ undefined: its denominator is 0" + (" ✅" if u is None else "")
            elif u is None:
                status = "⬜ not entered"
            elif abs(u - 100 * t) <= 1.0:
                status, n_correct = "✅ correct", n_correct + 1
            else:
                status = f"❌ re-check: {m} = {FORMULAS[m]}"
            rows.append({"Measure": m, "Your answer (%)": u, "Check": status})
        db_exec(
            "INSERT INTO reports(ts, grp, scenario, a, b, c, d, se, sp, ppv, npv, true_prev, app_prev, n_correct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now(), group, scenario, a, b, c, d, *[nan_to_none(truth[m]) for m in METRICS], n_correct),
        )
        st.session_state[f"feedback::{scenario}"] = {"rows": rows, "n_correct": n_correct}

    fb = st.session_state.get(f"feedback::{scenario}")
    if fb:
        st.success(f"Submitted! Group {group}'s table for {scenario} is now on the class board (Step 3).")
        st.dataframe(pd.DataFrame(fb["rows"]), hide_index=True, width="stretch")
        if any(r["Check"].startswith("❌") for r in fb["rows"]):
            st.info("Fix any ❌ on paper and resubmit. Your counts are already on the class board.")

        logged = db_query(
            "SELECT diseased, test_pos FROM draws WHERE grp = ? AND scenario = ?", (group, scenario)
        )
        with st.expander("🔍 Compare with the app's record of your group's animals"):
            if logged.empty:
                st.write("The app has no recorded draws for your group in this scenario.")
            else:
                la = int(((logged.diseased == 1) & (logged.test_pos == 1)).sum())
                lb = int(((logged.diseased == 0) & (logged.test_pos == 1)).sum())
                lc = int(((logged.diseased == 1) & (logged.test_pos == 0)).sum())
                ld = int(((logged.diseased == 0) & (logged.test_pos == 0)).sum())
                st.caption("Every animal your group sampled in Step 1:")
                render_2x2(la, lb, lc, ld)
                st.caption(
                    "If this differs from your paper table, an animal was miscounted or a member "
                    "sampled extra animals. Worth a quick check!"
                )


def class_results():
    st.title("📊 Step 3 · Class results")
    st.markdown(
        "Every group used **the same test on the same population**. "
        "Look at how much the estimates differ. The bars are 95% confidence intervals."
    )
    scenario = scenario_picker(label="Scenario")
    revealed = params_revealed()
    truth = expected_metrics(SCENARIOS[scenario])

    @st.fragment(run_every="15s")
    def board():
        rep = latest_reports(scenario)
        st.progress(
            min(len(rep) / N_GROUPS, 1.0),
            text=f"{len(rep)} of {N_GROUPS} groups have reported for {scenario} · refreshes every 15 s",
        )
        if rep.empty:
            st.info("No group has reported yet. Results will appear here automatically.")
            return

        # ---- table of each group's results
        tbl = rep[["grp", "a", "b", "c", "d"]].copy()
        tbl["n"] = tbl[["a", "b", "c", "d"]].sum(axis=1)
        for col, m in zip(["se", "sp", "ppv", "npv", "true_prev", "app_prev"], METRICS):
            tbl[m] = rep[col] * 100
        A, B, C, D = (int(rep[k].sum()) for k in "abcd")
        pooled = {"grp": "Class", "a": A, "b": B, "c": C, "d": D, "n": A + B + C + D}
        pooled.update({m: v * 100 for m, v in two_by_two_metrics(A, B, C, D).items()})
        tbl = pd.concat([tbl.astype({"grp": str}), pd.DataFrame([pooled])], ignore_index=True)
        tbl["grp"] = tbl["grp"].apply(lambda g: g if g == "Class" else f"Group {g}")
        st.dataframe(
            tbl.rename(columns={"grp": "Group"}),
            hide_index=True,
            width="stretch",
            column_config={m: st.column_config.NumberColumn(m, format="%.1f%%") for m in METRICS},
        )

        # ---- forest-style plot
        rows = []
        sources = [(f"Group {r.grp}", r.a, r.b, r.c, r.d) for r in rep.itertuples()]
        sources.append(("Class (pooled)", A, B, C, D))
        for name, a, b, c, d in sources:
            for m, (x, n) in metric_parts(a, b, c, d).items():
                lo, hi = wilson_ci(x, n)
                rows.append({"Source": name, "Measure": m, "Estimate": safe_div(x, n),
                             "lo": lo, "hi": hi, "x": x, "n": n,
                             "Truth": truth[m] if revealed else None})
        long = pd.DataFrame(rows)
        order = [s[0] for s in sources]
        base = alt.Chart().encode(y=alt.Y("Source:N", sort=order, title=None))
        ci = base.mark_rule(strokeWidth=3, opacity=0.6).encode(
            x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            x2="hi:Q",
        )
        pts = base.mark_point(filled=True, size=90).encode(
            x="Estimate:Q",
            color=alt.condition(alt.datum.Source == "Class (pooled)", alt.value("#d62728"), alt.value("#1f77b4")),
            tooltip=["Source", "Measure", alt.Tooltip("Estimate:Q", format=".1%"),
                     alt.Tooltip("x:Q", title="numerator"), alt.Tooltip("n:Q", title="denominator"),
                     alt.Tooltip("lo:Q", format=".1%", title="95% CI low"),
                     alt.Tooltip("hi:Q", format=".1%", title="95% CI high")],
        )
        layers = [ci, pts]
        if revealed:
            layers.append(alt.Chart().mark_rule(color="green", strokeDash=[5, 4], strokeWidth=2).encode(x="Truth:Q"))
        chart = (
            alt.layer(*layers, data=long)
            .properties(width=260, height=max(120, 24 * len(order)))
            .facet(facet=alt.Facet("Measure:N", sort=METRICS, title=None), columns=3)
        )
        st.altair_chart(chart)
        st.caption(
            "🔵 each group · 🔴 whole class pooled"
            + (" · green dashed line = true value" if revealed else "")
            + ". Hover over a point to see its numerator and denominator."
        )

        # ---- across scenarios
        allrep = latest_reports()
        if allrep["scenario"].nunique() > 1:
            st.markdown("##### Same test, different populations: pooled class results by scenario")
            out = []
            for scen, g in allrep.groupby("scenario"):
                a, b, c, d = (int(g[k].sum()) for k in "abcd")
                row = {"Scenario": scen, "Groups": len(g), "n": a + b + c + d}
                row.update({m: v * 100 for m, v in two_by_two_metrics(a, b, c, d).items()})
                out.append(row)
            st.dataframe(
                pd.DataFrame(out), hide_index=True, width="stretch",
                column_config={m: st.column_config.NumberColumn(m, format="%.1f%%") for m in METRICS},
            )

    board()

    if revealed:
        show_true_params(scenario)

    with st.expander("💬 Discuss with your group", expanded=True):
        st.markdown(
            """
1. Every group used the *same* test on the *same* population. **Why do your estimates differ?**
2. Which measure has the **widest** confidence intervals? Look at its **denominator**:
   in a low-prevalence scenario, how many truly diseased animals did your group actually see?
3. How close is the **pooled class** estimate to your group's? What would happen with 1,000 animals?
4. If you've done two scenarios: the **test is identical**, so why do PPV and NPV change?

➡️ Questions 2–3 lead to **Step 4 · Does sample size matter?** and question 4 to **Step 5**.
"""
        )


def sample_size():
    st.title("📏 Step 4 · Does sample size matter?")
    st.markdown(
        "Your group's estimates were based on only a handful of animals. "
        "Here the computer does the sampling so you can try **much bigger studies** and "
        "**repeat** the same study hundreds of times."
    )
    scenario = scenario_picker(label="Scenario", key="ss_scenario")
    p = SCENARIOS[scenario]
    revealed = params_revealed()
    truth = expected_metrics(p)
    rng = np.random.default_rng()

    tab1, tab2 = st.tabs(["A · One bigger study", "B · Repeat the study many times"])

    with tab1:
        n = st.select_slider("Number of animals to sample and test", [25, 50, 100, 200, 500, 1000, 5000], value=100)
        if st.button("🧪 Sample & test", type="primary"):
            D = rng.binomial(n, p["prevalence"])
            a = int(rng.binomial(D, p["sensitivity"]))
            d = int(rng.binomial(n - D, p["specificity"]))
            st.session_state["batch"] = (scenario, n, a, (n - D) - d, D - a, d)
        batch = st.session_state.get("batch")
        if batch and batch[0] == scenario:
            _, n_b, a, b, c, d = batch
            c1, c2 = st.columns([2, 3], gap="large")
            with c1:
                st.markdown(f"**Results for {n_b:,} animals**")
                render_2x2(a, b, c, d)
            with c2:
                rows = []
                for m, (x, den) in metric_parts(a, b, c, d).items():
                    lo, hi = wilson_ci(x, den)
                    rows.append({"Measure": m, "Estimate": pct(safe_div(x, den)),
                                 "95% CI": f"{pct(lo)} – {pct(hi)}" if den else "undefined",
                                 "Based on": f"{x} / {den}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Click again: how much do the numbers move? Now try a larger sample.")

    with tab2:
        c1, c2, c3 = st.columns(3)
        ns = c1.multiselect("Study sizes to compare", [10, 25, 50, 100, 250, 500, 1000, 2500], default=[25, 100, 500])
        reps = c2.slider("Repeat each study this many times", 100, 2000, 500, step=100)
        measure = c3.selectbox("Measure to look at", METRICS)
        if st.button("▶️ Run the repeated studies", type="primary") and ns:
            recs = []
            for n in sorted(ns):
                D = rng.binomial(n, p["prevalence"], reps)
                a = rng.binomial(D, p["sensitivity"])
                d = rng.binomial(n - D, p["specificity"])
                b, c = (n - D) - d, D - a
                num, den = metric_parts(a, b, c, d)[measure]
                est = np.where(den > 0, num / np.maximum(den, 1), np.nan)
                recs.append(pd.DataFrame({"n": n, "Estimate": est}))
            st.session_state["repeat"] = (scenario, measure, pd.concat(recs, ignore_index=True))

        res = st.session_state.get("repeat")
        if res and res[0] == scenario:
            _, meas, sims = res
            sims = sims.assign(Truth=truth[meas] if revealed else None)
            hist = alt.Chart().mark_bar(opacity=0.8).encode(
                x=alt.X("Estimate:Q", bin=alt.Bin(extent=[0, 1], step=0.02),
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=f"Estimated {meas}"),
                y=alt.Y("count():Q", title="Number of studies"),
            )
            layers = [hist]
            if revealed:
                layers.append(alt.Chart().mark_rule(color="green", strokeDash=[5, 4], strokeWidth=2).encode(x="Truth:Q"))
            chart = alt.layer(*layers, data=sims.dropna(subset=["Estimate"])).properties(width=650, height=110).facet(
                row=alt.Row("n:O", title="Animals per study")
            )
            st.altair_chart(chart)

            summ = []
            for n, g in sims.groupby("n"):
                e = g["Estimate"].dropna()
                lo, hi = (np.percentile(e, [2.5, 97.5]) if len(e) else (np.nan, np.nan))
                summ.append({"Animals per study": n,
                             "Studies where it couldn't be calculated": f"{g['Estimate'].isna().mean():.0%}",
                             "Middle 95% of estimates": f"{pct(lo)} – {pct(hi)}",
                             "Spread (width)": pct(hi - lo)})
            st.dataframe(pd.DataFrame(summ), hide_index=True, width="stretch")
            st.info(
                "💡 Each bar chart shows the answers you'd get from many identical studies. "
                "Notice how the spread narrows roughly with the **square root** of the sample size, and that "
                "sensitivity depends on how many **diseased** animals you happen to catch, not on the total *n*."
            )
    if revealed:
        show_true_params(scenario)


def predictive_values():
    st.title("🎯 Step 5 · Prevalence and predictive values")
    st.markdown(
        "Sensitivity and specificity belong to the **test**. PPV and NPV also depend on **who you test**. "
        "Start from the two examples in the lecture, then explore."
    )
    preset = st.radio(
        "Start from", ["Brucellosis: common (25%)", "Bovine TB: rare (0.1%)", "Custom"], horizontal=True
    )
    default_prev = {"Brucellosis: common (25%)": 25.0, "Bovine TB: rare (0.1%)": 0.1, "Custom": 10.0}[preset]
    c1, c2, c3 = st.columns(3)
    se = c1.slider("Sensitivity", 0.50, 1.00, 0.99, 0.005, key=f"pv_se::{preset}")
    sp = c2.slider("Specificity", 0.50, 1.00, 0.99, 0.005, key=f"pv_sp::{preset}")
    prev = c3.number_input("True prevalence (%)", 0.01, 99.99, default_prev, step=0.1,
                           format="%.2f", key=f"pv_prev::{preset}") / 100

    N = 100_000
    D = N * prev
    a, c = round(D * se), round(D) - round(D * se)
    d = round((N - D) * sp)
    b = (N - round(D)) - d
    m = two_by_two_metrics(a, b, c, d)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(f"**Expected results when testing {N:,} animals**")
        render_2x2(a, b, c, d)
        k1, k2 = st.columns(2)
        k1.metric("PPV", pct(m["PPV"]))
        k2.metric("NPV", pct(m["NPV"], 2))
        if a + b:
            per100 = 100 * a / (a + b)
            st.markdown(
                f"Out of every **100 positive** results, about **{per100:.0f}** are true positives and "
                f"**{100 - per100:.0f}** are false positives."
            )

    with right:
        grid = np.logspace(-4, np.log10(0.995), 300)
        curve = pd.DataFrame(
            [{"Prevalence": x, "Value": v, "Measure": k}
             for x in grid
             for k, v in (("PPV", se * x / (se * x + (1 - sp) * (1 - x))),
                          ("NPV", sp * (1 - x) / (sp * (1 - x) + (1 - se) * x)))]
        )
        lines = alt.Chart(curve).mark_line(strokeWidth=3).encode(
            x=alt.X("Prevalence:Q", scale=alt.Scale(type="log"), axis=alt.Axis(format="%"), title="True prevalence (log scale)"),
            y=alt.Y("Value:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["PPV", "NPV"], range=["#d62728", "#1f77b4"])),
        )
        now_rule = alt.Chart(pd.DataFrame({"x": [prev]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((lines + now_rule).properties(height=330, title="PPV and NPV across prevalence"),
                        width="stretch")

    st.divider()
    st.subheader("Apparent vs true prevalence")
    st.markdown(
        "Screening tells you the **apparent** prevalence (proportion testing positive). If Se and Sp are known, "
        "the Rogan–Gladen estimator corrects it:  **true prevalence = (AP + Sp − 1) / (Se + Sp − 1)**. "
        "*Lecture example: 1.5% of UC Davis students test positive for COVID-19 with a 99%/99% test.*"
    )
    r1, r2, r3 = st.columns(3)
    ap = r1.number_input("Apparent prevalence (%)", 0.0, 100.0, 1.5, 0.1) / 100
    se2 = r2.number_input("Test sensitivity", 0.5, 1.0, 0.99, 0.01)
    sp2 = r3.number_input("Test specificity", 0.5, 1.0, 0.99, 0.01)
    if se2 + sp2 - 1 <= 0:
        st.error("Se + Sp must be greater than 1 for the test to be informative.")
    else:
        tp = (ap + sp2 - 1) / (se2 + sp2 - 1)
        if tp < 0:
            st.warning(
                f"Estimated true prevalence is below 0 ({pct(tp)}): fewer positives than the "
                f"{pct(1 - sp2)} false-positive rate alone would produce. Report it as ~0%."
            )
        elif tp > 1:
            st.warning("Estimate is above 100%. Check Se and Sp.")
        else:
            st.metric("Estimated true prevalence", pct(tp, 2), delta=f"{100 * (tp - ap):+.2f} points vs apparent",
                      delta_color="off")


def thresholds_roc():
    st.title("🐕 Step 6 · Cut-offs and ROC curves")
    st.markdown(
        """
Many tests give a **number**, not a yes/no. Here you'll diagnose **canine hypothyroidism** from total
thyroxine (**T4**) or triiodothyronine (**T3**). Hypothyroid dogs *tend* to have lower values, but the groups overlap.
You choose the cut-off: dogs **below** it are called *test positive*.
"""
    )
    group = st.session_state.get("group")
    c1, c2, c3 = st.columns(3)
    breed = c1.radio("Breed", list(BREED_PREVALENCE))
    test_type = c2.radio("Hormone test", ["T4", "T3"], help="For the class exercise, start with T4.")
    n = c3.slider("Dogs to sample", 30, 200, 60, step=10)
    if st.button("🐕 Sample dogs", type="primary"):
        vals = [generate_dog(test_type, breed) for _ in range(n)]
        st.session_state["dogs"] = {
            "breed": breed, "test": test_type,
            "df": pd.DataFrame(vals, columns=["value", "hypo"]),
        }
    dogs = st.session_state.get("dogs")
    if not dogs:
        st.info("Choose a breed and a test, then click **Sample dogs**.")
        return

    df = dogs["df"].copy()
    tt, br = dogs["test"], dogs["breed"]
    lo, hi, unit = NORMAL_RANGE[tt]
    st.caption(f"Showing **{len(df)} {br}s** tested with **{tt}**. Change the options and sample again for a new population.")

    thr = st.slider(f"Cut-off: call a dog hypothyroid if {tt} is below … ({unit})",
                    0.0, 50.0 if tt == "T4" else 100.0, lo, 0.5, key=f"thr::{tt}")
    df["Status"] = np.where(df["hypo"], "Hypothyroid", "Healthy")
    df["Result"] = np.where(df["value"] < thr, "Positive", "Negative")
    tp = int(((df.hypo) & (df.Result == "Positive")).sum())
    fp = int(((~df.hypo) & (df.Result == "Positive")).sum())
    fn = int(((df.hypo) & (df.Result == "Negative")).sum())
    tn = int(((~df.hypo) & (df.Result == "Negative")).sum())
    m = two_by_two_metrics(tp, fp, fn, tn)

    left, right = st.columns([3, 2], gap="large")
    with left:
        pts = alt.Chart(df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("Status:N", title=None, sort=["Hypothyroid", "Healthy"]),
            xOffset="jitter:Q",
            y=alt.Y("value:Q", title=f"{tt} ({unit})"),
            color=alt.Color("Result:N", title="Test result",
                            scale=alt.Scale(domain=["Positive", "Negative"], range=["#d62728", "#1f77b4"])),
            tooltip=[alt.Tooltip("value:Q", title=tt), "Status", "Result"],
        ).transform_calculate(jitter="random()")
        normal = alt.Chart(pd.DataFrame({"y": [lo, hi]})).mark_rule(color="green", strokeDash=[5, 5]).encode(y="y:Q")
        cut = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(color="black", strokeWidth=2).encode(y="y:Q")
        st.altair_chart((pts + normal + cut).properties(height=420), width="stretch")
        st.caption(f"Black line = your cut-off · green dashed = reference interval ({lo:g}–{hi:g} {unit})")
    with right:
        render_2x2(tp, fp, fn, tn)
        k1, k2, k3 = st.columns(3)
        k1.metric("Sensitivity", pct(m["Sensitivity"]))
        k2.metric("Specificity", pct(m["Specificity"]))
        k3.metric("Youden's J", f"{m['Sensitivity'] + m['Specificity'] - 1:.2f}"
                  if not (math.isnan(m["Sensitivity"]) or math.isnan(m["Specificity"])) else "undefined")
        k4, k5 = st.columns(2)
        k4.metric("PPV", pct(m["PPV"]))
        k5.metric("NPV", pct(m["NPV"]))
        if group:
            if st.button(f"📤 Report this cut-off for Group {group}", width="stretch"):
                db_exec(
                    "INSERT INTO thresholds(ts, grp, breed, test_type, n, threshold, tp, fp, fn, tn, se, sp) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (now(), group, br, tt, len(df), thr, tp, fp, fn, tn,
                     nan_to_none(m["Sensitivity"]), nan_to_none(m["Specificity"])),
                )
                st.toast("Cut-off reported to the class board ✔️")
        else:
            st.caption("Choose your group in the sidebar to report cut-offs to the class.")
        st.caption("Try at least **two** cut-offs, report each, and note how Se and Sp trade off.")

    st.divider()
    st.subheader(f"Class board: every cut-off reported for {tt}")

    @st.fragment(run_every="20s")
    def threshold_board():
        th = db_query("SELECT * FROM thresholds WHERE test_type = ? ORDER BY id", (tt,))
        if th.empty:
            st.info("No cut-offs reported yet.")
            return
        th = th.dropna(subset=["se", "sp"]).assign(fpr=lambda x: 1 - x.sp, Group=lambda x: "Group " + x.grp.astype(str))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        sc = alt.Chart(th).mark_point(filled=True, size=90).encode(
            x=alt.X("fpr:Q", scale=alt.Scale(domain=[0, 1]), title="1 − Specificity (false-positive proportion)"),
            y=alt.Y("se:Q", scale=alt.Scale(domain=[0, 1]), title="Sensitivity"),
            color=alt.Color("breed:N", title="Breed"),
            shape=alt.Shape("Group:N"),
            tooltip=["Group", "breed", alt.Tooltip("threshold:Q", title="cut-off"),
                     alt.Tooltip("se:Q", format=".0%"), alt.Tooltip("sp:Q", format=".0%"), "n"],
        )
        st.altair_chart((diag + sc).properties(height=380), width="stretch")
        st.caption(
            "Each point is one group's cut-off. Together, the points trace out an **ROC curve**. "
            "Points near the top-left corner are the best trade-offs; the dashed diagonal is a coin toss."
        )

    threshold_board()

    st.divider()
    st.subheader("ROC curve for your sample")
    if get_setting("reveal_roc") != "1":
        st.info("🔒 Your instructor will unlock this section after groups have reported their cut-offs.")
        return
    y = df["hypo"].to_numpy()
    if y.all() or (~y).all():
        st.warning("Your sample contains only one disease status, so an ROC curve can't be drawn. Sample more dogs.")
        return
    vals = df["value"].to_numpy()
    cuts = np.unique(np.concatenate([[0.0], vals + 1e-6, [vals.max() + 1]]))
    se_c = np.array([((vals < t) & y).sum() / y.sum() for t in cuts])
    fpr_c = np.array([((vals < t) & ~y).sum() / (~y).sum() for t in cuts])
    auc = float(np.sum(np.diff(fpr_c) * (se_c[1:] + se_c[:-1]) / 2))
    j = se_c - fpr_c
    best = int(np.argmax(j))
    roc = pd.DataFrame({"fpr": fpr_c, "se": se_c, "cut": cuts})
    line = alt.Chart(roc).mark_line(strokeWidth=3, color="#ff7f0e", interpolate="step-after").encode(
        x=alt.X("fpr:Q", title="1 − Specificity"), y=alt.Y("se:Q", title="Sensitivity"),
        tooltip=[alt.Tooltip("cut:Q", format=".1f", title="cut-off"), alt.Tooltip("se:Q", format=".0%"),
                 alt.Tooltip("fpr:Q", format=".0%", title="1 − Sp")],
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    star = alt.Chart(roc.iloc[[best]]).mark_point(shape="diamond", size=200, filled=True, color="red").encode(x="fpr:Q", y="se:Q")
    c1, c2 = st.columns([3, 2], gap="large")
    c1.altair_chart((diag + line + star).properties(height=380), width="stretch")
    with c2:
        st.metric("Area under the curve (AUC)", f"{auc:.2f}")
        st.metric("Best cut-off by Youden's J", f"below {cuts[best]:.1f} {unit}")
        st.write(f"Sensitivity **{pct(se_c[best])}**, specificity **{pct(1 - fpr_c[best])}**, J = **{j[best]:.2f}**")
        st.markdown("**True breed prevalence in this simulation**")
        st.dataframe(pd.DataFrame({"Breed": list(BREED_PREVALENCE),
                                   "Prevalence": [pct(v, 0) for v in BREED_PREVALENCE.values()]}),
                     hide_index=True)
        st.caption(
            "Prevalence changes PPV and NPV but **not** the ROC curve. Compare breeds at the same cut-off. "
            "And is the 'best' cut-off by J always the best clinically? Think about the cost of a false negative."
        )


def likelihood_ratios():
    st.title("⚖️ Step 7 · Likelihood ratios")
    st.markdown(
        """
Likelihood ratios combine sensitivity and specificity into one number that does **not** depend on prevalence,
and let you update **your patient's** probability of disease:

**Post-test odds = pre-test odds × LR**  where  LR+ = Se / (1 − Sp)  and  LR− = (1 − Se) / Sp
"""
    )
    c1, c2, c3 = st.columns(3)
    se = c1.slider("Sensitivity", 0.50, 0.999, 0.80, 0.01, key="lr_se")
    sp = c2.slider("Specificity", 0.50, 0.999, 0.90, 0.01, key="lr_sp")
    pre = c3.slider("Pre-test probability (your clinical judgement)", 0.01, 0.99, 0.10, 0.01, key="lr_pre")

    lr_pos, lr_neg = se / (1 - sp), (1 - se) / sp
    pre_odds = pre / (1 - pre)
    post_pos_odds, post_neg_odds = pre_odds * lr_pos, pre_odds * lr_neg
    post_pos, post_neg = post_pos_odds / (1 + post_pos_odds), post_neg_odds / (1 + post_neg_odds)

    k = st.columns(4)
    k[0].metric("LR+", f"{lr_pos:.2f}")
    k[1].metric("LR−", f"{lr_neg:.2f}")
    k[2].metric("Probability if test +", pct(post_pos), delta=f"{100 * (post_pos - pre):+.0f} pts")
    k[3].metric("Probability if test −", pct(post_neg), delta=f"{100 * (post_neg - pre):+.0f} pts")

    with st.expander("Show the working (positive test)", expanded=True):
        st.markdown(
            f"""
1. Pre-test odds = {pre:.2f} / (1 − {pre:.2f}) = **{pre_odds:.3f}**
2. Post-test odds = {pre_odds:.3f} × LR+ {lr_pos:.2f} = **{post_pos_odds:.3f}**
3. Post-test probability = {post_pos_odds:.3f} / (1 + {post_pos_odds:.3f}) = **{post_pos:.2f}**
"""
        )
    verdict_pos = ("strongly **rules in** disease" if lr_pos > 10 else
                   "moderately raises the probability" if lr_pos > 2 else "barely changes the probability")
    verdict_neg = ("strongly **rules out** disease" if lr_neg < 0.1 else
                   "moderately lowers the probability" if lr_neg < 0.5 else "barely changes the probability")
    st.markdown(f"A positive result {verdict_pos}; a negative result {verdict_neg}.")

    grid = np.linspace(0.001, 0.999, 200)
    odds = grid / (1 - grid)
    cdf = pd.DataFrame({
        "Pre-test": np.tile(grid, 3),
        "Post-test": np.concatenate([odds * lr_pos / (1 + odds * lr_pos), odds * lr_neg / (1 + odds * lr_neg), grid]),
        "Result": ["Test positive"] * 200 + ["Test negative"] * 200 + ["No test"] * 200,
    })
    lines = alt.Chart(cdf).mark_line(strokeWidth=3).encode(
        x=alt.X("Pre-test:Q", axis=alt.Axis(format="%"), title="Pre-test probability"),
        y=alt.Y("Post-test:Q", axis=alt.Axis(format="%"), title="Post-test probability"),
        color=alt.Color("Result:N", scale=alt.Scale(domain=["Test positive", "Test negative", "No test"],
                                                    range=["#d62728", "#1f77b4", "#aaaaaa"])),
        strokeDash=alt.condition(alt.datum.Result == "No test", alt.value([4, 4]), alt.value([1, 0])),
    )
    here = alt.Chart(pd.DataFrame({"x": [pre, pre], "y": [post_pos, post_neg]})).mark_point(
        size=120, filled=True, color="black").encode(x="x:Q", y="y:Q")
    st.altair_chart((lines + here).properties(height=360), width="stretch")
    st.caption("A test is most useful when it moves you across a decision threshold (to treat, or to test further).")

    st.divider()
    st.subheader("Quick check")
    picked = st.multiselect("Which of these depend on the prevalence in the population tested?",
                            ["Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR−"])
    if st.button("Check my answer"):
        if set(picked) == {"PPV", "NPV"}:
            st.success("✅ Right: only the predictive values depend on prevalence. Se, Sp and the LRs are properties of the test.")
        else:
            st.error("Not quite. Which measures change when you test a different population with the same test?")


def instructor():
    st.title("🔐 Instructor panel")
    try:
        password = st.secrets["INSTRUCTOR_PASSWORD"]
    except Exception:
        password = os.environ.get("INSTRUCTOR_PASSWORD", "epi205")
    if not st.session_state.get("is_instructor"):
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
            if pw == password:
                st.session_state["is_instructor"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        if password == "epi205":
            st.caption("⚠️ Using the default password. Set INSTRUCTOR_PASSWORD in your app secrets before class.")
        return

    st.subheader("Class controls")
    c1, c2 = st.columns(2)
    with c1:
        names = list(SCENARIOS)
        cur = get_setting("active_scenario")
        active = st.selectbox("Active scenario (default shown to students)", names,
                              index=names.index(cur) if cur in names else 0)
        cap = st.number_input("Single-sampling draws per student", 1, 50, int(get_setting("draws_per_student")))
    with c2:
        rp = st.toggle("Reveal true scenario parameters to students", get_setting("reveal_params") == "1")
        rr = st.toggle("Unlock ROC curve / Youden answer (Step 6)", get_setting("reveal_roc") == "1")
    if st.button("💾 Save settings", type="primary"):
        set_setting("active_scenario", active)
        set_setting("draws_per_student", int(cap))
        set_setting("reveal_params", int(rp))
        set_setting("reveal_roc", int(rr))
        st.success("Saved. Students see the changes on their next interaction.")

    st.markdown("**True scenario parameters**")
    st.dataframe(pd.DataFrame(SCENARIOS).T.map(lambda v: f"{v:.0%}"), width="stretch")

    st.subheader("Live progress")
    draws = db_query("SELECT * FROM draws")
    reports = db_query("SELECT * FROM reports")
    ths = db_query("SELECT * FROM thresholds")
    p1, p2, p3 = st.columns(3)
    p1.metric("Animals sampled", len(draws))
    p2.metric("2×2 reports", len(reports))
    p3.metric("Cut-offs reported", len(ths))
    if not draws.empty:
        st.markdown("Animals sampled, by group and scenario")
        st.dataframe(pd.crosstab(draws["grp"], draws["scenario"]).rename_axis("Group"), width="stretch")
    if not reports.empty:
        st.markdown("Latest 2×2 reports (n_correct = calculations the group got right, out of 6)")
        st.dataframe(latest_reports(), hide_index=True, width="stretch")

    st.subheader("Download data")
    d1, d2, d3 = st.columns(3)
    stamp = datetime.now().strftime("%Y%m%d")
    d1.download_button("⬇️ Draws (CSV)", draws.to_csv(index=False), f"draws_{stamp}.csv", "text/csv")
    d2.download_button("⬇️ 2×2 reports (CSV)", reports.to_csv(index=False), f"reports_{stamp}.csv", "text/csv")
    d3.download_button("⬇️ Cut-offs (CSV)", ths.to_csv(index=False), f"thresholds_{stamp}.csv", "text/csv")

    st.subheader("Reset")
    tables = st.multiselect("Clear these tables", ["draws", "reports", "thresholds"])
    confirm = st.checkbox("I have downloaded what I need and want to delete this data")
    if st.button("🗑️ Clear selected data", disabled=not (tables and confirm)):
        for t in tables:
            db_exec(f"DELETE FROM {t}")
        st.success(f"Cleared: {', '.join(tables)}")

    if st.button("Log out"):
        st.session_state["is_instructor"] = False
        st.rerun()


# --------------------------------------------------------------------------------------
# Sidebar + navigation
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👥 Your group")
    st.selectbox(
        "Group number",
        [None] + list(range(1, N_GROUPS + 1)),
        format_func=lambda g: "Select…" if g is None else f"Group {g}",
        key="group",
    )
    st.text_input("Your initials (optional)", key="student", max_chars=10)
    st.divider()

pages = {
    "Start here": [st.Page(home, title="Welcome & instructions", icon="🏠", default=True)],
    "Part 1 · Build your 2×2 table": [
        st.Page(single_sampling, title="1. Sample animals", icon="🎲"),
        st.Page(group_report, title="2. Report your 2×2", icon="📝"),
        st.Page(class_results, title="3. Class results", icon="📊"),
    ],
    "Part 2 · Going further": [
        st.Page(sample_size, title="4. Does sample size matter?", icon="📏"),
        st.Page(predictive_values, title="5. Prevalence & predictive values", icon="🎯"),
        st.Page(thresholds_roc, title="6. Cut-offs & ROC curves", icon="🐕"),
        st.Page(likelihood_ratios, title="7. Likelihood ratios", icon="⚖️"),
    ],
    "Instructor": [st.Page(instructor, title="Instructor panel", icon="🔐")],
}
st.navigation(pages).run()
