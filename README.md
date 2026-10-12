# Properties of Diagnostic Tests: Epi 205 class activity

One Streamlit app that replaces the separate `disease_simulation.py`, `disease_simulation_batch.py`
and `ROC_curve_testing*.py` apps. Students move through seven steps in the sidebar, and every
group's results are pooled on a live class board.

| Step | Page | Replaces / adds |
|---|---|---|
| 0 | Welcome & instructions | Learning objectives, session roadmap, blank 2×2 template |
| 1 | Sample animals | `disease_simulation.py`: shows true status + test result; students place each animal in a cell (instant feedback). Draws per student are capped (default 5). |
| 2 | Report your 2×2 | **New.** Group enters a, b, c, d and their six calculations; the app checks each one and posts the table to the class board. |
| 3 | Class results | **New.** Live table + forest plot of every group's Se, Sp, PPV, NPV, true and apparent prevalence with 95% Wilson CIs, pooled class estimate, and a cross-scenario comparison. Auto-refreshes every 15 s. |
| 4 | Does sample size matter? | `disease_simulation_batch.py` (any n) + repeated-study simulation showing how the spread of estimates shrinks with n |
| 5 | Prevalence & predictive values | Brucellosis (25%) / bovine TB (0.1%) examples from the lecture, PPV/NPV-vs-prevalence curve, Rogan–Gladen true-prevalence correction (COVID example) |
| 6 | Cut-offs & ROC curves | `ROC_curve_testing.py`: strip plot with movable cut-off; groups report cut-offs and their points build an ROC curve on a shared board. ROC/AUC/Youden answer unlocks from the instructor panel (was `ROC_curve_testingAnswers.py`). |
| 7 | Likelihood ratios | LR+, LR−, pre-test to post-test probability with worked steps (matches the lecture example: 0.10 → 0.47) and a prevalence-dependence quick check |
| | Instructor panel | Password protected: set the active scenario, draws per student, reveal true parameters, unlock ROC answers, live progress, CSV downloads, reset |

## Suggested class flow

1. **Steps 1–3 (≈ 40 min).** Instructor sets Scenario 1 as active. Each student samples 5 animals,
   the group (≈ 4–5 students, so ≈ 20–25 animals) builds one 2×2 table on paper, one person submits it.
   Project the **Class results** page. At 10% prevalence each group sees only ~2 diseased animals,
   so the sensitivity intervals are huge and groups disagree. That's the hook.
2. Switch the active scenario to **Scenario 2** (same test, 90% prevalence) and repeat. The class board's
   cross-scenario table shows PPV/NPV swinging while Se/Sp stay put.
3. Toggle **Reveal true parameters**, discuss, then send groups to **Steps 4–5**.
4. **Step 6:** each group reports at least two T4 cut-offs; the shared board turns into an ROC curve.
   Unlock the ROC answer when ready. **Step 7** wraps up with likelihood ratios.
5. Download the CSVs from the instructor panel, then reset before the next section.

## Running it

**Streamlit Community Cloud:** push this folder to GitHub, create an app with main file `app.py`,
and paste `INSTRUCTOR_PASSWORD = "..."` into *Advanced settings → Secrets*.

**Locally / Codespaces:** `pip install -r requirements.txt` then `streamlit run app.py`
(the included `.devcontainer` does this automatically). For a local password, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Without one, the password is `epi205`.

## About the shared data

Class results live in `class_results.db` (SQLite) next to `app.py` and are shared by all students
connected to the same running app. On Community Cloud the file survives while the app is running
but **is wiped when the app reboots, sleeps, or is redeployed**. So avoid pushing code changes
during class, and download the CSVs at the end of each session. If you need results to persist
across weeks, swap the four small `db_*` helper functions for a Google Sheet or a hosted database.

Scenario parameters (`SCENARIOS`), breed prevalences (`BREED_PREVALENCE`) and the number of groups
(`N_GROUPS = 8`) are set at the top of `app.py`.
