# PlayerPulse: Predicting Which Players Will Quit Next Month

**A churn model for an online game, built end to end: raw activity logs → features → imbalanced classification → subgroup performance audit → tested API on the web.**

**Live API:** https://playerpulse-api.vercel.app (opens the interactive docs: try `POST /predict` in the browser)

The data is the **World of Warcraft Avatar History** dataset: real activity logs from one WoW server in 2008. Every 10 minutes the server recorded which characters were online, their level, class, race, guild and zone. That gives **10.8 million snapshots for 37,354 characters**.

---

## Results

| | Result |
|---|---|
| **Who is scored** | Engaged players (5+ hours in the month): 34,779 player-months |
| **How many churn** | About **14%** do not log in the next month (imbalanced) |
| **Ranking quality (test, Sep–Nov)** | **PR-AUC 0.60** (random = 0.14) · ROC-AUC 0.89 |
| **At the chosen threshold** | Catches **73% of churners** at **48% precision** |
| **Retention budget** | Contacting the riskiest **10%** of players each month catches **45% of churners**, with 65% of contacts being real churners (**4.5× better than random**) |
| **Imbalance** | At the default 0.5 threshold recall was only 34–38%; class weights, undersampling or a tuned threshold raised it to 65–81% |
| **Load test (live API)** | **3,380 players scored per second** with batch requests, 0 errors in 3,400 requests |
| **Tests** | 14 pytest tests, including one that checks the API matches the trained model to 0.001; run by GitHub Actions on every push |

---

## Method

**1. Player-month table** (`src/step1_build_dataset.py`)
One row per character per month: hours played, active days, levels gained, zones visited, share of time in cities / dungeons / battlegrounds / arenas (from `zones.csv`), guild activity, days since last seen, tenure, and the change from last month.
**Label:** `churned = 1` if the character is not seen at all in the next calendar month.

Overall 43% of player-months churn, but most of that is players who logged in for under an hour (69% churn). Those are not the players a live-ops team acts on, so the model scores **players with 5+ hours in the month**, where churn is 14%.

| Hours played in the month | Player-months | Churn next month |
|---|---|---|
| < 1 h | 39,357 | 69% |
| 1–5 h | 17,055 | 43% |
| 5–20 h | 13,797 | 24% |
| 20–50 h | 9,543 | 10% |
| 50 h + | 11,107 | 4% |

**2. Models and imbalance** (`src/step2_train.py`)
Split by time so the model never sees the future: **train Jan–Jul, validate Aug, test Sep–Nov**.
Compared logistic regression and XGBoost, each with no weighting, class weights, and undersampling, at threshold 0.5 and at a threshold tuned on August. The final model was chosen on the **validation** month, not the test months.

| Model | Imbalance handling | Threshold | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic | none | 0.5 | 0.617 | 0.750 | 0.338 | 0.465 |
| Logistic | class weights | tuned | 0.620 | 0.557 | 0.647 | 0.599 |
| XGBoost | none | 0.5 | 0.599 | 0.683 | 0.380 | 0.488 |
| **XGBoost (chosen)** | none | **tuned (0.20)** | 0.599 | 0.480 | **0.731** | 0.580 |
| XGBoost | undersampling | 0.5 | 0.572 | 0.412 | 0.808 | 0.546 |

**Why XGBoost, when logistic regression scores higher on test?** (`src/step6_selection_budget.py`)

| Model | Imbalance handling | Validation PR-AUC (Aug) | Test PR-AUC (Sep–Nov) |
|---|---|---|---|
| **XGBoost (chosen)** | none | **0.683** | 0.599 |
| XGBoost | class weights | 0.676 | 0.601 |
| XGBoost | undersampling | 0.664 | 0.572 |
| Logistic | none | 0.640 | 0.617 |
| Logistic | class weights | 0.640 | 0.620 |
| Logistic | undersampling | 0.639 | 0.620 |

The choice has to be made before seeing the test months, and on August XGBoost was clearly best (0.683 vs 0.640). On the test months the order flipped: XGBoost dropped by 0.08, logistic by only 0.02. The test months include the November 2008 expansion (new class, new level cap), and the simpler linear model held up better under that shift. Picking logistic now because of the test result would be tuning on the test set, so the choice stays, and the result is reported as it is. **Takeaway:** when the game is about to change a lot, a simpler model plus a retrain after the update is the safer bet.

**Choosing who to contact.** The 0.20 threshold maximises F1 on August, but a retention team usually has a fixed budget. Ranking each month's players by churn risk (test months, churn rate 14.4%):

| Strategy | Players contacted | Churners caught | Share of all churners | Precision | False alarms | vs random |
|---|---|---|---|---|---|---|
| Top 5% each month | 440 | 328 | 26% | 75% | 112 | 5.2× |
| **Top 10% each month** | 880 | 569 | **45%** | 65% | 311 | 4.5× |
| Top 20% each month | 1,762 | 896 | 71% | 51% | 866 | 3.5× |
| Threshold 0.20 (F1-tuned) | 1,931 | 927 | 73% | 48% | 1,004 | 3.3× |

Going from the top 20% to the threshold adds 169 contacts but catches only 31 more churners. If an offer is cheap (an in-game reward), the threshold is fine; if it is costly (a free month), the top 5–10% gives far more churners per offer.

The strongest signal by far is **days since last seen**, then active days, level and guild activity.

**3. Subgroup performance audit** (`src/step3_fairness.py`)
Recall (share of real churners caught) and false-positive rate by in-game race, class and level band on the test months. These are game attributes, not real-world demographics, so this checks **which kinds of players the model serves badly**, not demographic fairness:

| Group | Recall range | Gap |
|---|---|---|
| Race (5 races) | 0.68 – 0.77 | 0.09 |
| Class (10 classes) | 0.44 (Death Knight) – 0.80 | 0.36 |
| Level band | 0.27 (71–80) – 0.82 (1–59) | 0.55 |

The big gaps are **level 71–80** and **Death Knights**. Both only exist after the *Wrath of the Lich King* expansion (November 2008), so **the model never saw them in training**. Per-level-band thresholds tuned on August barely changed the gap (0.55 → 0.53) and cost precision (0.48 → 0.38), because a threshold cannot fix a group the model has never seen. The right fix is to detect new groups and retrain, so the API **warns** when a player's class or values are outside the training data.

**4. API** (`api/main.py`, FastAPI)

| Endpoint | Returns |
|---|---|
| `POST /predict` | churn probability, yes/no at the threshold, risk level, warnings |
| `POST /predict/batch` | up to 500 players at once |
| `GET /model` | model, threshold, test metrics |
| `GET /fairness` | subgroup audit: recall / false-positive rate by race, class, level band |
| `GET /health` | status |

**5. Load test** (`src/step5_load_test.py`, run from India against the live Vercel API)

| Endpoint | Requests in flight | Requests | Requests / s | Players / s | p50 | p95 | Errors |
|---|---|---|---|---|---|---|---|
| `/predict` | 10 | 200 | 32 | 32 | 290 ms | 460 ms | 0 |
| `/predict` | 50 | 1,000 | 152 | 152 | 290 ms | 513 ms | 0 |
| `/predict` | 100 | 2,000 | 53 | 53 | 1,313 ms | 5,293 ms | 0 |
| `/predict/batch` (100 players each) | 20 | 200 | 34 | **3,380** | 497 ms | 733 ms | 0 |

Most of the ~290 ms is the network round trip from India to the server; scoring itself takes a few milliseconds. Single requests top out around 150 per second on the free Vercel plan and slow down at 100 in flight, so **batching is how to reach thousands of predictions per second**. No request failed.

The API loads the exported XGBoost trees (`models/booster.json`) with plain NumPy scaling, so it does not need scikit-learn or pandas. `tests/test_api.py` checks input validation, warnings, batch limits, and that its probabilities match the full training pipeline.

---

## Run it

```bash
pip install -r requirements-train.txt
# put wowah_data.csv and zones.csv from Kaggle in data/raw/
python src/step1_build_dataset.py
python src/step2_train.py
python src/step3_fairness.py
python src/step4_export.py
python src/step6_selection_budget.py   # model choice + budget table

pip install -r requirements-dev.txt
python -m pytest -q
uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs
python src/step5_load_test.py https://playerpulse-api.vercel.app
```

## Limitations

- **One server, one year (2008).** Behaviour on a modern game would differ.
- **"Not seen next month" is a proxy for churn.** Some players come back later.
- **New content breaks the model** (see subgroup audit): it needs retraining after big updates.
- **Free-tier hosting:** Vercel's free plan limits single-request throughput; drift monitoring and an Azure deployment are planned next.

---

**Tech:** Python · pandas · scikit-learn · XGBoost · FastAPI · pytest · GitHub Actions · Vercel

**Data:** [World of Warcraft Avatar History](https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history), Kaggle (Lee et al., 2011)

**Author:** Nalluri Sanjay, B.Tech (Hons.) Industrial & Systems Engineering, IIT Kharagpur · [LinkedIn](https://www.linkedin.com/in/sanjaynalluri2405/) · [GitHub](https://github.com/SanjayNalluri)
