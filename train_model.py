import json, os, glob
import numpy as np
import tensorflow as tf

os.makedirs("output/ranknet", exist_ok=True)

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

# ── Load data ─────────────────────────────────────────────────────────────────

LB_PATH = "data/lb.json"
with open(LB_PATH, "r", encoding="utf-8") as f:
    lb_data = json.load(f)

student_files = [
    f for f in glob.glob("data/*.json")
    if os.path.basename(f).lower() != "lb.json"
]
print(f"Leaderboard: {LB_PATH}")
print(f"Student files ({len(student_files)}): {[os.path.basename(f) for f in student_files]}")

# ── N lookup & test metadata lookup ───────────────────────────────────────────

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

        if test.get("maxMarks") and name not in max_marks_lookup:
            max_marks_lookup[name] = test["maxMarks"]
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
print(f"N lookup built for {len(n_lookup)} tests | avg N: {avg_N:.0f}")

# ── Build training points ─────────────────────────────────────────────────────

points = []
point_test_names = []
point_scores = []
point_ranks = []

# Index lb_tests for O(1) lookup
lb_tests = lb_data.get("tests", []) if isinstance(lb_data, dict) else lb_data
lb_by_name = {t.get("testName"): t for t in lb_tests if t.get("testName")}

seen_keys = set()

# Process Leaderboard points from lb.json
for test in lb_tests:
    name   = test.get("testName")
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
print(f"Unique Leaderboard points: {lb_count}")

# Process ALL student and converted dataset files in data/
for sf in student_files:
    with open(sf, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        continue

    for test in data:
        score = test.get("score")
        rank  = test.get("rank")
        pct   = test.get("percentile")
        if score is None or score <= 0 or rank is None or rank <= 0:
            continue

        name     = test.get("testName")
        if not name: continue
        maxMarks = test.get("maxMarks") or max_marks_lookup.get(name)
        if not maxMarks or maxMarks <= 0: continue

        # Extract avg & topper from record itself, lb_tests, or global lookups
        lb = lb_by_name.get(name)
        avg = test.get("avg") or (lb.get("avg") if lb else None) or avg_lookup.get(name)
        topper = test.get("topper") or (lb.get("topper") if lb else None) or topper_lookup.get(name)
        if not avg or not topper or topper <= avg:
            continue

        # Determine N from lookup or directly from percentile
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

print(f"Additional unique points from all data/*.json files: {len(points) - lb_count}")
print(f"Total deduplicated training points: {len(points)}")

points = np.array(points, dtype=np.float32)
point_test_names = np.array(point_test_names)
point_scores = np.array(point_scores)
point_ranks = np.array(point_ranks)

X = points[:, :4]
Y = points[:, 4]

print(f"\nz range:          {X[:,0].min():.3f} → {X[:,0].max():.3f}")
print(f"x_norm range:     {X[:,1].min():.3f} → {X[:,1].max():.3f}")
print(f"difficulty range: {X[:,2].min():.3f} → {X[:,2].max():.3f}")
print(f"maxMarks range:   {X[:,3].min():.3f} → {X[:,3].max():.3f}")
print(f"y range:          {(Y.min()*100):.1f}% → {(Y.max()*100):.1f}%")

# ── Model (Tuned for sharper tail resolution & smooth leaderboard inversion) ──

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(48, activation="tanh"),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(1,  activation="sigmoid")
], name="ranknet")

# Original training process preserved (Adam 1e-3, 2500 epochs, batch size 32)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

print("\nTraining RankNet (2500 epochs, batch_size=32)...")
model.fit(X, Y, epochs=2500, batch_size=32, verbose=0)
print("Training complete!")

# ── Comprehensive Evaluation & Per-Test Analytics ─────────────────────────────

preds  = model.predict(X, verbose=0).flatten()
errors = np.abs(preds - Y) * 100

overall_mae    = float(np.mean(errors))
overall_median = float(np.median(errors))
overall_rmse   = float(np.sqrt(np.mean(errors ** 2)))
overall_max    = float(np.max(errors))
worst_idx      = int(np.argmax(errors))

w_1  = float((errors < 1.0).mean() * 100)
w_2  = float((errors < 2.0).mean() * 100)
w_3  = float((errors < 3.0).mean() * 100)
w_5  = float((errors < 5.0).mean() * 100)
w_10 = float((errors < 10.0).mean() * 100)

print("\n" + "="*70)
print("                       OVERALL EVALUATION RESULTS                       ")
print("="*70)
print(f"Average Error (MAE):     {overall_mae:.2f} percentile pts")
print(f"Median Error:            {overall_median:.2f} percentile pts")
print(f"RMSE:                    {overall_rmse:.2f} percentile pts")
print(f"Max Absolute Error:      {overall_max:.2f} percentile pts")
print(f"Within ±1.0 percentile:  {w_1:.1f}%")
print(f"Within ±2.0 percentile:  {w_2:.1f}%")
print(f"Within ±3.0 percentile:  {w_3:.1f}%")
print(f"Within ±5.0 percentile:  {w_5:.1f}%")
print(f"Within ±10.0 percentile: {w_10:.1f}%")
print(f"\nSingle Worst Point: Test '{point_test_names[worst_idx]}' | Score: {point_scores[worst_idx]} | Rank: {point_ranks[worst_idx]}")
print(f"  Actual: {Y[worst_idx]*100:.2f}% | Predicted: {preds[worst_idx]*100:.2f}% | Error: {errors[worst_idx]:.2f} pts")

# Per-Test Error Breakdown
unique_tests = np.unique(point_test_names)
test_stats = []

for t_name in unique_tests:
    mask = (point_test_names == t_name)
    t_err = errors[mask]
    test_stats.append({
        "testName": t_name,
        "sampleCount": int(len(t_err)),
        "mae": float(np.mean(t_err)),
        "median": float(np.median(t_err)),
        "maxError": float(np.max(t_err)),
        "within_3pts": float((t_err < 3).mean() * 100),
        "within_5pts": float((t_err < 5).mean() * 100)
    })

# Sort by MAE
test_stats.sort(key=lambda x: x["mae"])
best_test = test_stats[0]
worst_test = test_stats[-1]

# Also find test with absolute maximum single error
test_with_max_peak_error = max(test_stats, key=lambda x: x["maxError"])

print("\n" + "="*70)
print("                         TEST-LEVEL BREAKDOWN                           ")
print("="*70)
print(f"Total Unique Tests Evaluated: {len(test_stats)}")
print(f"\n★ BEST PREDICTED TEST (Lowest MAE):")
print(f"  '{best_test['testName']}'")
print(f"  MAE: {best_test['mae']:.2f} pts | Max: {best_test['maxError']:.2f} pts | Within ±3pts: {best_test['within_3pts']:.1f}% ({best_test['sampleCount']} samples)")

print(f"\n▲ WORST PREDICTED TEST (Highest MAE):")
print(f"  '{worst_test['testName']}'")
print(f"  MAE: {worst_test['mae']:.2f} pts | Max: {worst_test['maxError']:.2f} pts | Within ±3pts: {worst_test['within_3pts']:.1f}% ({worst_test['sampleCount']} samples)")

print(f"\n▲ TEST WITH HIGHEST PEAK ERROR:")
print(f"  '{test_with_max_peak_error['testName']}'")
print(f"  Peak Error: {test_with_max_peak_error['maxError']:.2f} pts | MAE: {test_with_max_peak_error['mae']:.2f} pts")

print("\n" + "-"*70)
print(f"{'TEST NAME':<44} | {'SAMPLES':<7} | {'MAE':<7} | {'MAX ERR':<7}")
print("-"*70)
print("Top 3 Best Tests:")
for t in test_stats[:3]:
    print(f"  {t['testName'][:42]:<42} | {t['sampleCount']:<7} | {t['mae']:<7.2f} | {t['maxError']:<7.2f}")
print("\nTop 3 Hardest Tests:")
for t in test_stats[-3:]:
    print(f"  {t['testName'][:42]:<42} | {t['sampleCount']:<7} | {t['mae']:<7.2f} | {t['maxError']:<7.2f}")
print("="*70)

# ── Save ──────────────────────────────────────────────────────────────────────

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

with open(f"output/ranknet/{shard_name}", "wb") as f:
    f.write(bin_bytes)

# Ensure InputLayer config has ONLY batch_input_shape (not both inputShape and batchInputShape)
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
                "name": "ranknet",
                "layers": layers
            }
        }
    },
    "weightsManifest": weights_manifest
}

with open("output/ranknet/model.json", "w", encoding="utf-8") as f:
    json.dump(model_json, f, indent=2)

meta = {
    "inputs":       ["z", "x_norm", "difficulty", "maxMarks_norm"],
    "output":       "percentile (0-1)",
    "avg_N":        avg_N,
    "maxMarks_ref": 300,
    "stat_constants": {
        "slope_tgn":     SLOPE_TGN,
        "intercept_tgn": INTERCEPT_TGN,
        "slope_k":       SLOPE_K,
        "intercept_k":   INTERCEPT_K
    },
    "evaluation": {
        "overall": {
            "mae_percentile_pts": round(overall_mae, 2),
            "median_pts":         round(overall_median, 2),
            "rmse_pts":           round(overall_rmse, 2),
            "max_error_pts":      round(overall_max, 2),
            "within_1pt":         round(w_1, 1),
            "within_2pts":        round(w_2, 1),
            "within_3pts":        round(w_3, 1),
            "within_5pts":        round(w_5, 1),
            "within_10pts":       round(w_10, 1)
        },
        "worst_test_by_mae": {
            "name": worst_test["testName"],
            "mae": round(worst_test["mae"], 2),
            "max_error": round(worst_test["maxError"], 2),
            "samples": worst_test["sampleCount"]
        },
        "best_test_by_mae": {
            "name": best_test["testName"],
            "mae": round(best_test["mae"], 2),
            "max_error": round(best_test["maxError"], 2),
            "samples": best_test["sampleCount"]
        },
        "test_with_max_peak_error": {
            "name": test_with_max_peak_error["testName"],
            "peak_error": round(test_with_max_peak_error["maxError"], 2),
            "mae": round(test_with_max_peak_error["mae"], 2)
        },
        "all_tests": test_stats
    }
}

with open("output/ranknet/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → output/ranknet/model.json & group1-shard1of1.bin")
print(f"Saved → output/ranknet/meta.json with full test breakdown")
