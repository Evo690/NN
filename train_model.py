import json, os, glob, re, csv
import numpy as np
import tensorflow as tf

os.makedirs("output/ranknet", exist_ok=True)
os.makedirs("output/ranknet_institute", exist_ok=True)
os.makedirs("output/ranknet_combined", exist_ok=True)

# ── Fitted statistical constants ──────────────────────────────────────────────

SLOPE_TGN     = -0.6776
INTERCEPT_TGN =  0.8130
SLOPE_K       = -4.6774
INTERCEPT_K   =  4.5740

# ── Statistical normalization ─────────────────────────────────────────────────

def estimate_topper(avg, maxMarks):
    difficulty      = avg / maxMarks
    topper_gap_norm = SLOPE_TGN * difficulty + INTERCEPT_TGN
    topper          = avg + topper_gap_norm * maxMarks
    return min(topper, maxMarks)

def dynamic_k(difficulty):
    return SLOPE_K * difficulty + INTERCEPT_K

def normalize_input(score, avg, maxMarks):
    difficulty    = avg / maxMarks
    topper        = estimate_topper(avg, maxMarks)
    k             = dynamic_k(difficulty)
    sigma         = (topper - avg) / k if k > 0 else 1.0
    z             = (score - avg) / sigma if sigma > 0 else 0.0
    gap           = topper - avg
    x_norm        = (score - avg) / gap if gap > 0 else 0.0
    x_norm        = max(0.01, min(x_norm, 1.0))
    maxMarks_norm = maxMarks / 300.0
    return z, x_norm, difficulty, maxMarks_norm

# ── Filter for Standard JEE tests ─────────────────────────────────────────────

def is_valid_jee_test(name, maxMarks=None):
    if not name:
        return False
    nl = name.strip().lower()

    # 1. Exclude Medical (NEET) tests
    if "neet" in nl:
        return False

    # 2. Exclude Admission / Scholarship tests (N-ASAT, ASAT, NTSC)
    if "asat" in nl or "ntsc" in nl:
        return False

    # 3. Exclude Olympiads (NSEP, NSEC, NSEA, IOQM, etc.)
    if any(k in nl for k in ["nsep", "nsec", "nsea", "inpho", "incho", "ioqm", "olympiad"]):
        return False

    # 4. Exclude junior foundation tests (Class 8, 9, 10, IX, X, SST)
    if any(k in nl for k in ["class 8", "class 9", "class 10", "class-8", "class-9", "class-10", "class 08", "class 09", "sst", "class ix", "_ix", " ix "]):
        return False

    # 5. Exclude Alpha & Beta drill tests (skewed extreme curves)
    if "alpha" in nl or "beta" in nl:
        return False

    # 6. Exclude early 2025 tests (IT 1 to IT 3 from 2025 with wrong/glitchy percentile data)
    if ("25" in nl or "2025" in nl):
        if re.search(r"(?:it|test)[\s\-_]*0?[123](?:[^\d]|$)", nl):
            return False

    # 7. Standard JEE marks range: 120 <= maxMarks <= 396 (Mains 300; Adv 180, 186, 198, 360)
    if maxMarks is not None:
        if maxMarks < 120 or maxMarks > 396:
            return False

    return True

# ── Load Institute Data ───────────────────────────────────────────────────────

LB_PATH = "data/lb.json"
with open(LB_PATH, "r", encoding="utf-8") as f:
    lb_data = json.load(f)

student_files = [
    f for f in glob.glob("data/*.json")
    if os.path.basename(f).lower() != "lb.json"
    and "checkthis" not in os.path.basename(f).lower()
    and "master" not in os.path.basename(f).lower()
]
print(f"Leaderboard: {LB_PATH}")
print(f"Student files ({len(student_files)}): {[os.path.basename(f) for f in student_files]}")

# N lookup & test metadata lookup (JEE Only)
n_lookup = {}
max_marks_lookup = {}
avg_lookup = {}
topper_lookup = {}

for sf in student_files:
    with open(sf, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        continue

    for test in data:
        name = test.get("testName")
        if not name:
            continue

        mm = test.get("maxMarks")
        if not is_valid_jee_test(name, mm):
            continue

        if mm and name not in max_marks_lookup:
            max_marks_lookup[name] = mm
        if test.get("avg") and name not in avg_lookup:
            avg_lookup[name] = test["avg"]
        if test.get("topper") and name not in topper_lookup:
            topper_lookup[name] = test["topper"]

        if not test.get("score")      or test["score"] == 0:      continue
        if not test.get("rank")       or test["rank"] == 0:       continue
        if not test.get("percentile") or test["percentile"] == 0: continue
        if test["percentile"] >= 100:                              continue

        N = test["rank"] / (1 - test["percentile"] / 100)
        if name not in n_lookup:
            n_lookup[name] = []
        n_lookup[name].append(N)

avg_N = float(np.mean([np.mean(vals) for vals in n_lookup.values()])) if n_lookup else 500.0
print(f"N lookup built for {len(n_lookup)} JEE tests | avg N: {avg_N:.0f}")

# Build Institute training points
points = []
point_test_names = []
point_scores = []
point_ranks = []

lb_tests = lb_data.get("tests", []) if isinstance(lb_data, dict) else lb_data
lb_by_name = {t.get("testName"): t for t in lb_tests if t.get("testName")}

seen_keys = set()

# Process Leaderboard points from lb.json (JEE only)
for test in lb_tests:
    name     = test.get("testName")
    maxMarks = max_marks_lookup.get(name) or test.get("maxMarks")
    if not is_valid_jee_test(name, maxMarks):
        continue

    avg    = test.get("avg") or avg_lookup.get(name)
    topper = test.get("topper") or topper_lookup.get(name)
    lb     = test.get("leaderboard", [])
    if not lb and isinstance(test.get("summary"), dict):
        lb = test["summary"].get("leaderboard", [])

    if not avg or not topper or not lb:    continue
    if topper <= avg:                      continue
    if name not in n_lookup:               continue
    if name not in max_marks_lookup:       continue

    N        = float(np.mean(n_lookup[name]))
    maxMarks = max_marks_lookup[name]

    for entry in lb:
        score = entry.get("score") if entry.get("score") is not None else entry.get("totalMarks")
        rank  = entry.get("rank") if entry.get("rank") is not None else entry.get("ranks")
        if score is None or rank is None or score <= 0 or rank <= 0: continue

        y = 1.0 - (rank / N)
        if not (0 < y < 1.0):             continue

        z, x_norm, difficulty, maxMarks_norm = normalize_input(score, avg, maxMarks)
        if x_norm <= 0:                    continue

        key = (name, round(float(score), 1), int(rank))
        if key in seen_keys: continue
        seen_keys.add(key)

        points.append([z, x_norm, difficulty, maxMarks_norm, y])
        point_test_names.append(name)
        point_scores.append(score)
        point_ranks.append(rank)

lb_count = len(points)
print(f"Unique JEE Leaderboard points: {lb_count}")

# Process ALL student and converted dataset files in data/ (JEE only)
for sf in student_files:
    with open(sf, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        continue

    for test in data:
        name = test.get("testName")
        if not name: continue
        maxMarks = test.get("maxMarks") or max_marks_lookup.get(name)
        if not is_valid_jee_test(name, maxMarks):
            continue

        score = test.get("score")
        rank  = test.get("rank")
        pct   = test.get("percentile")
        if score is None or score <= 0 or rank is None or rank <= 0:
            continue

        lb = lb_by_name.get(name)
        avg = test.get("avg") or (lb.get("avg") if lb else None) or avg_lookup.get(name)
        topper = test.get("topper") or (lb.get("topper") if lb else None) or topper_lookup.get(name)
        if not avg or not topper or topper <= avg:
            continue

        n_vals = n_lookup.get(name)
        if n_vals:
            N = float(np.mean(n_vals))
        elif pct and 0 < pct < 100:
            N = float(rank) / (1.0 - float(pct) / 100.0)
        else:
            continue

        y = 1.0 - (rank / N)
        if not (0 < y < 1.0):
            continue

        z, x_norm, difficulty, maxMarks_norm = normalize_input(score, avg, maxMarks)
        if x_norm <= 0:
            continue

        key = (name, round(float(score), 1), int(rank))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        points.append([z, x_norm, difficulty, maxMarks_norm, y])
        point_test_names.append(name)
        point_scores.append(score)
        point_ranks.append(rank)

points = np.array(points, dtype=np.float32)
point_test_names = np.array(point_test_names)
point_scores = np.array(point_scores)
point_ranks = np.array(point_ranks)

X_inst = points[:, :4]
Y_inst = points[:, 4]

print(f"Total Institute JEE training points: {len(X_inst)} across {len(np.unique(point_test_names))} tests")

# ── Process checkthis.csv (Filter: Score >= 100, remove suspicious accounts) ──

checkthis_points = []
CHECKTHIS_PATH = "checkthis.csv"
if not os.path.exists(CHECKTHIS_PATH):
    CHECKTHIS_PATH = os.path.join("data", "checkthis.csv")

if os.path.exists(CHECKTHIS_PATH):
    candidates = []
    all_scores = []
    max_rank = 0
    filter_stats = {
        "score_lt_100": 0,
        "score_gt_275": 0,
        "acc_100": 0,
        "acc_unrealistic": 0,
        "negative_marks": 0,
        "skipped_subject": 0,
        "subject_skew_70pct": 0
    }
    with open(CHECKTHIS_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                s = float(row["total_marks"])
                r = int(row["rank"])
                all_scores.append(s)
                if r > max_rank:
                    max_rank = r

                acc = float(row.get("accuracy", 0))
                m_m = float(row.get("mathematics_marks", 0))
                p_m = float(row.get("physics_marks", 0))
                c_m = float(row.get("chemistry_marks", 0))

                # 1. Minimum threshold: score >= 100 (user specified)
                if s < 100.0:
                    filter_stats["score_lt_100"] += 1
                    continue
                # 2. Exclude extreme top suspected cheaters (> 275 marks)
                if s > 275.0:
                    filter_stats["score_gt_275"] += 1
                    continue
                # 3. Filter out unrealistic / leaked accuracy (100% or >= 96% when score >= 180)
                if acc >= 100.0:
                    filter_stats["acc_100"] += 1
                    continue
                if acc >= 96.0 and s >= 180.0:
                    filter_stats["acc_unrealistic"] += 1
                    continue
                # 4. Filter out negative marks in any subject for 100+ scorers
                if min(m_m, p_m, c_m) < 0:
                    filter_stats["negative_marks"] += 1
                    continue
                # 5. Filter out skipped subjects (e.g. min <= 5 and max >= 50 marks)
                if min(m_m, p_m, c_m) <= 5.0 and max(m_m, p_m, c_m) >= 50.0:
                    filter_stats["skipped_subject"] += 1
                    continue
                # 6. Filter out weird subject distribution (> 70% of score from single subject)
                if (max(m_m, p_m, c_m) / s) > 0.70:
                    filter_stats["subject_skew_70pct"] += 1
                    continue

                candidates.append({"rank": r, "score": s})
            except Exception:
                pass

    candidates.sort(key=lambda x: x["rank"])
    ct_avg = float(np.mean(all_scores)) if all_scores else 102.66
    ct_maxMarks = 300.0
    ct_N = float(max_rank) if max_rank > 0 else 5358.0

    print(f"\n[checkthis.csv Conversion & Filter Report]")
    print(f"  - Total Candidates (N): {ct_N:.0f} | Exam Batch Avg: {ct_avg:.2f} | Max Marks: {ct_maxMarks:.0f}")
    print(f"  - Excluded score < 100: {filter_stats['score_lt_100']}")
    print(f"  - Excluded score > 275 (extreme cheaters): {filter_stats['score_gt_275']}")
    print(f"  - Excluded 100% accuracy: {filter_stats['acc_100']}")
    print(f"  - Excluded unrealistic accuracy (>=96% at 180+ marks): {filter_stats['acc_unrealistic']}")
    print(f"  - Excluded negative subject marks: {filter_stats['negative_marks']}")
    print(f"  - Excluded skipped subjects (<=5 in one, >=50 in another): {filter_stats['skipped_subject']}")
    print(f"  - Excluded severe subject skew (>70% from 1 subject): {filter_stats['subject_skew_70pct']}")
    print(f"  [OK] Clean Serious Candidates Remaining: {len(candidates)} (Ranks {candidates[0]['rank']} to {candidates[-1]['rank']})")

    # Stratified subsampling: 150 points evenly spaced across the clean candidates
    if len(candidates) > 150:
        indices = np.linspace(0, len(candidates) - 1, 150, dtype=int)
        sampled_candidates = [candidates[i] for i in indices]
    else:
        sampled_candidates = candidates

    for item in sampled_candidates:
        s = item["score"]
        r = item["rank"]
        y = 1.0 - (r / ct_N)
        if 0 < y < 1.0:
            z, x_norm, diff, max_norm = normalize_input(s, ct_avg, ct_maxMarks)
            if x_norm > 0:
                checkthis_points.append([z, x_norm, diff, max_norm, y])

    print(f"[checkthis.csv] Subsampled {len(checkthis_points)} clean representative points for Model B")

# Build Combined Dataset (Institute + Cleaned Checkthis)
if checkthis_points:
    checkthis_arr = np.array(checkthis_points, dtype=np.float32)
    X_comb = np.vstack([X_inst, checkthis_arr[:, :4]])
    Y_comb = np.concatenate([Y_inst, checkthis_arr[:, 4]])
else:
    X_comb, Y_comb = X_inst, Y_inst

print(f"Dataset summary: Model A = {len(X_inst)} samples | Model B = {len(X_comb)} samples")

# ── Model Factory ─────────────────────────────────────────────────────────────

def build_ranknet(name):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(48, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(16, activation="tanh"),
        tf.keras.layers.Dense(1,  activation="sigmoid")
    ], name=name)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return m

# ── Train Model A (Institute Data Only) ────────────────────────────────────────

print("\n" + "="*70)
print("TRAINING MODEL A (Baseline: Institute Data Only - 2500 epochs, batch 32)...")
print("="*70)
model_a = build_ranknet("ranknet_institute")
model_a.fit(X_inst, Y_inst, epochs=2500, batch_size=32, verbose=0)
print("Model A training complete!")

# ── Train Model B (Combined: Institute + Cleaned Checkthis) ───────────────────

print("\n" + "="*70)
print("TRAINING MODEL B (Combined: Institute + Checkthis - 2500 epochs, batch 32)...")
print("="*70)
model_b = build_ranknet("ranknet_combined")
model_b.fit(X_comb, Y_comb, epochs=2500, batch_size=32, verbose=0)
print("Model B training complete!")

# ── Comparative Benchmark on Institute Data ───────────────────────────────────

# ── Model Evaluation Helper ───────────────────────────────────────────────────

def print_detailed_model_stats(model_label, preds, targets, test_names, X):
    errors = np.abs(preds - targets) * 100
    mae = float(np.mean(errors))
    median = float(np.median(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_err = float(np.max(errors))
    w1 = float((errors < 1.0).mean() * 100)
    w3 = float((errors < 3.0).mean() * 100)
    w5 = float((errors < 5.0).mean() * 100)

    unique_tests = np.unique(test_names)
    test_stats = []
    for t in unique_tests:
        m = (test_names == t)
        t_errs = errors[m]
        test_stats.append({
            "name": t,
            "samples": int(np.sum(m)),
            "mae": float(np.mean(t_errs)),
            "max": float(np.max(t_errs)),
            "w3": float((t_errs < 3.0).mean() * 100)
        })

    test_stats_by_mae = sorted(test_stats, key=lambda x: x["mae"])
    best_test = test_stats_by_mae[0]
    worst_test = test_stats_by_mae[-1]

    test_stats_by_peak = sorted(test_stats, key=lambda x: x["max"], reverse=True)
    peak_test = test_stats_by_peak[0]

    worst_idx = int(np.argmax(errors))

    print("\n" + "=" * 78)
    print(f"             DETAILED PERFORMANCE STATS: {model_label.upper()}             ")
    print("=" * 78)
    print(f"Total Unique Tests Evaluated: {len(unique_tests)} | Total Test Samples: {len(targets)}")
    print(f"  * Average Error (MAE) : {mae:.2f} percentile pts")
    print(f"  * Median Absolute Err : {median:.2f} percentile pts")
    print(f"  * Root Mean Sq Error  : {rmse:.2f} percentile pts")
    print(f"  * Max Peak Error      : {max_err:.2f} percentile pts")
    print(f"  * Within +/-1.0 pts   : {w1:.1f}%")
    print(f"  * Within +/-3.0 pts   : {w3:.1f}%")
    print(f"  * Within +/-5.0 pts   : {w5:.1f}%")

    print(f"\n★ BEST PREDICTED TEST (Lowest MAE):")
    print(f"  '{best_test['name']}'")
    print(f"  MAE: {best_test['mae']:.2f} pts | Max: {best_test['max']:.2f} pts | Within +/-3pts: {best_test['w3']:.1f}% ({best_test['samples']} samples)")

    print(f"\n▲ WORST PREDICTED TEST (Highest MAE):")
    print(f"  '{worst_test['name']}'")
    print(f"  MAE: {worst_test['mae']:.2f} pts | Max: {worst_test['max']:.2f} pts | Within +/-3pts: {worst_test['w3']:.1f}% ({worst_test['samples']} samples)")

    print(f"\n▲ TEST WITH HIGHEST PEAK ERROR:")
    print(f"  '{peak_test['name']}'")
    print(f"  Peak Error: {peak_test['max']:.2f} pts | MAE: {peak_test['mae']:.2f} pts ({peak_test['samples']} samples)")

    print("\n" + "-" * 78)
    print(f"{'TEST NAME':<46} | {'SAMPLES':<7} | {'MAE':<6} | {'MAX ERR':<7} | {'+/-3PTS'}")
    print("-" * 78)
    print("Top 3 Best Predicted Tests:")
    for t in test_stats_by_mae[:3]:
        t_display = t['name'][:44]
        print(f"  {t_display:<44} | {t['samples']:<7} | {t['mae']:<6.2f} | {t['max']:<7.2f} | {t['w3']:<5.1f}%")

    print("\nTop 3 Worst Predicted Tests:")
    for t in test_stats_by_mae[-3:][::-1]:
        t_display = t['name'][:44]
        print(f"  {t_display:<44} | {t['samples']:<7} | {t['mae']:<6.2f} | {t['max']:<7.2f} | {t['w3']:<5.1f}%")
    print("-" * 78)

    print(f"\nWorst Individual Prediction Sample:")
    print(f"  Test: {test_names[worst_idx]}")
    print(f"  Inputs: z={X[worst_idx,0]:.3f}, x_norm={X[worst_idx,1]:.3f}, diff={X[worst_idx,2]:.3f}, maxMarks_norm={X[worst_idx,3]:.3f}")
    print(f"  Actual: {targets[worst_idx]*100:.2f}% | Predicted: {preds[worst_idx]*100:.2f}% | Error: {errors[worst_idx]:.2f} pts")
    print("=" * 78)

    return {
        "mae": round(mae, 2),
        "median": round(median, 2),
        "rmse": round(rmse, 2),
        "max": round(max_err, 2),
        "w1": round(w1, 1),
        "w3": round(w3, 1),
        "w5": round(w5, 1),
        "test_stats": test_stats
    }

# ── Evaluate Both Models On Institute Test Data ──────────────────────────────

preds_a = model_a.predict(X_inst, verbose=0).flatten()
preds_b = model_b.predict(X_inst, verbose=0).flatten()

stats_a = print_detailed_model_stats("Model A (OG Institute Data)", preds_a, Y_inst, point_test_names, X_inst)
stats_b = print_detailed_model_stats("Model B (OG Data + Checkthis)", preds_b, Y_inst, point_test_names, X_inst)

# ── Head-to-Head Comparative Benchmark ────────────────────────────────────────

err_a = np.abs(preds_a - Y_inst) * 100
err_b = np.abs(preds_b - Y_inst) * 100

mae_a, mae_b = stats_a["mae"], stats_b["mae"]
med_a, med_b = stats_a["median"], stats_b["median"]
rmse_a, rmse_b = stats_a["rmse"], stats_b["rmse"]
max_a, max_b = stats_a["max"], stats_b["max"]
w1_a, w1_b = stats_a["w1"], stats_b["w1"]
w3_a, w3_b = stats_a["w3"], stats_b["w3"]
w5_a, w5_b = stats_a["w5"], stats_b["w5"]

winner_mae = "MODEL A (Institute)" if mae_a < mae_b else "MODEL B (Combined)"
winner_rmse = "MODEL A (Institute)" if rmse_a < rmse_b else "MODEL B (Combined)"
winner_max = "MODEL A (Institute)" if max_a < max_b else "MODEL B (Combined)"

print("\n" + "=" * 85)
print("         HEAD-TO-HEAD BENCHMARK: EVALUATING BOTH ON INSTITUTE DATA      ")
print("=" * 85)
print(f"{'METRIC':<26} | {'MODEL A (INSTITUTE ONLY)':<25} | {'MODEL B (COMBINED)':<20} | {'BETTER'}")
print("-" * 85)
print(f"{'Average Error (MAE)':<26} | {mae_a:<25.2f} | {mae_b:<20.2f} | {winner_mae}")
print(f"{'Median Error':<26} | {med_a:<25.2f} | {med_b:<20.2f} | {'MODEL A' if med_a < med_b else 'MODEL B'}")
print(f"{'RMSE':<26} | {rmse_a:<25.2f} | {rmse_b:<20.2f} | {winner_rmse}")
print(f"{'Max Absolute Error':<26} | {max_a:<25.2f} | {max_b:<20.2f} | {winner_max}")
print(f"{'Within +/-1.0 percentile':<26} | {w1_a:<24.1f}% | {w1_b:<19.1f}% | {'MODEL A' if w1_a > w1_b else 'MODEL B'}")
print(f"{'Within +/-3.0 percentile':<26} | {w3_a:<24.1f}% | {w3_b:<19.1f}% | {'MODEL A' if w3_a > w3_b else 'MODEL B'}")
print(f"{'Within +/-5.0 percentile':<26} | {w5_a:<24.1f}% | {w5_b:<19.1f}% | {'MODEL A' if w5_a > w5_b else 'MODEL B'}")
print("=" * 85)

# Per-Test Breakdown
unique_tests = np.unique(point_test_names)
test_comparison = []

wins_a = 0
wins_b = 0

for t_name in unique_tests:
    mask = (point_test_names == t_name)
    m_err_a = float(np.mean(err_a[mask]))
    m_err_b = float(np.mean(err_b[mask]))
    if m_err_a < m_err_b:
        wins_a += 1
    else:
        wins_b += 1
    test_comparison.append({
        "testName": t_name,
        "sampleCount": int(np.sum(mask)),
        "mae_a": round(m_err_a, 2),
        "mae_b": round(m_err_b, 2),
        "winner": "A" if m_err_a < m_err_b else "B"
    })

print(f"\nTest-by-Test Winner Breakdown (out of {len(unique_tests)} tests):")
print(f"  * Model A won: {wins_a} tests ({(wins_a/len(unique_tests)*100):.1f}%)")
print(f"  * Model B won: {wins_b} tests ({(wins_b/len(unique_tests)*100):.1f}%)")

# Determine Best Overall Model
chosen_model = model_a if mae_a <= mae_b else model_b
chosen_name = "Model A (Institute Only)" if mae_a <= mae_b else "Model B (Combined)"
print(f"\n>>> PRIMARY MODEL SELECTED FOR DEPLOYMENT: {chosen_name} <<<")

# ── Save Helper ───────────────────────────────────────────────────────────────

def export_tfjs(model, target_dir, model_name):
    os.makedirs(target_dir, exist_ok=True)
    weights_manifest = []
    weight_specs = []
    bin_bytes = bytearray()
    for layer in model.layers:
        for w in layer.weights:
            base_name = w.name.split(":")[0]
            w_name = f"{layer.name}/{base_name}" if "/" not in base_name else base_name
            arr = w.numpy()
            weight_specs.append({"name": w_name, "shape": list(arr.shape), "dtype": "float32"})
            bin_bytes.extend(arr.astype("<f4").tobytes())

    shard_name = "group1-shard1of1.bin"
    weights_manifest.append({"paths": [shard_name], "weights": weight_specs})

    model_config = model.get_config()
    layers = model_config.get("layers", [])

    if layers and layers[0].get("class_name") != "InputLayer":
        layers.insert(0, {
            "class_name": "InputLayer",
            "config": {
                "batch_input_shape": [None, 4],
                "dtype": "float32",
                "sparse": False,
                "name": "input_1"
            }
        })

    for layer_obj in layers:
        cfg = layer_obj.get("config", {})
        if layer_obj.get("class_name") == "InputLayer":
            b_shape = cfg.get("batch_input_shape") or cfg.get("batch_shape") or [None, 4]
            cfg.clear()
            cfg["batch_input_shape"] = b_shape
            cfg["dtype"] = "float32"
            cfg["sparse"] = False
            cfg["name"] = "input_1"

    model_json = {
        "format": "layers-model",
        "generatedBy": "keras v2.15.0",
        "convertedBy": "TensorFlow.js Exporter",
        "modelTopology": {
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": model_name,
                    "layers": layers
                }
            }
        },
        "weightsManifest": weights_manifest
    }

    with open(os.path.join(target_dir, shard_name), "wb") as f:
        f.write(bin_bytes)
    with open(os.path.join(target_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)

# Export primary chosen model to output/ranknet/ and web_model/
export_tfjs(chosen_model, "output/ranknet", "ranknet")
export_tfjs(chosen_model, "web_model", "ranknet")

# Also save both individual models for reference/artifacts
export_tfjs(model_a, "output/ranknet_institute", "ranknet_institute")
export_tfjs(model_b, "output/ranknet_combined", "ranknet_combined")

meta = {
    "name": "RankNet Model Benchmark & Deployment Engine",
    "selected_model": chosen_name,
    "stat_constants": {
        "slope_tgn":     SLOPE_TGN,
        "intercept_tgn": INTERCEPT_TGN,
        "slope_k":       SLOPE_K,
        "intercept_k":   INTERCEPT_K
    },
    "benchmark": {
        "model_a_institute": {
            "training_samples": len(X_inst),
            "mae": round(mae_a, 2),
            "median": round(med_a, 2),
            "rmse": round(rmse_a, 2),
            "max_error": round(max_a, 2),
            "within_3pts": round(w3_a, 1),
            "within_5pts": round(w5_a, 1)
        },
        "model_b_combined": {
            "training_samples": len(X_comb),
            "checkthis_samples_added": len(checkthis_points),
            "mae": round(mae_b, 2),
            "median": round(med_b, 2),
            "rmse": round(rmse_b, 2),
            "max_error": round(max_b, 2),
            "within_3pts": round(w3_b, 1),
            "within_5pts": round(w5_b, 1)
        },
        "tests_won": {
            "model_a": wins_a,
            "model_b": wins_b
        },
        "all_tests": test_comparison
    }
}

for d in ["output/ranknet", "output/ranknet_institute", "output/ranknet_combined", "output", "web_model"]:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

print("\nSuccessfully saved all models and comparative benchmark metadata!")
print(f"  - Deployed best model to output/ranknet/ ({chosen_name})")
print(f"  - Preserved Model A in output/ranknet_institute/")
print(f"  - Preserved Model B in output/ranknet_combined/")
