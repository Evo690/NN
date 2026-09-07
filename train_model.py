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

# Leaderboard points
lb_tests = lb_data.get("tests", []) if isinstance(lb_data, dict) else lb_data
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
        if score is None or rank is None:  continue

        y = 1.0 - (rank / N)
        if not (0 < y < 1.0):             continue

        z, x_norm, difficulty, maxMarks_norm = normalize_input(score, avg, maxMarks)
        if x_norm <= 0:                    continue

        points.append([z, x_norm, difficulty, maxMarks_norm, y])

lb_count = len(points)
print(f"Leaderboard points: {lb_count}")

# Student personal points
for sf in student_files:
    with open(sf, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        continue

    for test in data:
        if not test.get("score")      or test["score"] == 0:      continue
        if not test.get("rank")       or test["rank"] == 0:       continue
        if not test.get("percentile") or test["percentile"] == 0: continue

        name     = test.get("testName")
        maxMarks = test.get("maxMarks") or max_marks_lookup.get(name)
        if not maxMarks:                                           continue

        lb = next((l for l in lb_tests if l.get("testName") == name), None)
        if not lb or not (lb.get("avg") or avg_lookup.get(name)) or not (lb.get("topper") or topper_lookup.get(name)):
            continue

        avg    = lb.get("avg") or avg_lookup.get(name)
        topper = lb.get("topper") or topper_lookup.get(name)
        n_vals = n_lookup.get(name)
        if not n_vals or topper <= avg:                            continue
        N      = float(np.mean(n_vals))

        y = 1.0 - (test["rank"] / N)
        if not (0 < y < 1.0):                                     continue

        z, x_norm, difficulty, maxMarks_norm = normalize_input(test["score"], avg, maxMarks)
        if x_norm <= 0:                                            continue

        points.append([z, x_norm, difficulty, maxMarks_norm, y])

print(f"Student personal points: {len(points) - lb_count}")
print(f"Total training points: {len(points)}")

points = np.array(points, dtype=np.float32)
X = points[:, :4]
Y = points[:, 4]

print(f"\nz range:          {X[:,0].min():.3f} → {X[:,0].max():.3f}")
print(f"x_norm range:     {X[:,1].min():.3f} → {X[:,1].max():.3f}")
print(f"difficulty range: {X[:,2].min():.3f} → {X[:,2].max():.3f}")
print(f"maxMarks range:   {X[:,3].min():.3f} → {X[:,3].max():.3f}")
print(f"y range:          {(Y.min()*100):.1f}% → {(Y.max()*100):.1f}%")

# ── Model ─────────────────────────────────────────────────────────────────────

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(1,  activation="sigmoid")
], name="ranknet")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

model.fit(X, Y, epochs=2500, batch_size=32, verbose=0)

# ── Evaluate ──────────────────────────────────────────────────────────────────

preds  = model.predict(X, verbose=0).flatten()
errors = np.abs(preds - Y) * 100

worst  = np.argmax(errors)

print(f"\nFinal MAE:    {errors.mean():.2f} percentile pts")
print(f"Max error:    {errors.max():.2f} percentile pts")
print(f"Within ±3pts: {(errors < 3).mean()*100:.1f}%")
print(f"Within ±5pts: {(errors < 5).mean()*100:.1f}%")
print(f"Worst — z:{X[worst,0]:.3f} x:{X[worst,1]:.3f} diff:{X[worst,2]:.3f} actual:{Y[worst]*100:.1f}% pred:{preds[worst]*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────

try:
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, "output/ranknet")
except Exception:
    weights_manifest = []
    weight_specs = []
    bin_bytes = bytearray()
    for layer in model.layers:
        for w in layer.weights:
            w_name = w.name.split(":")[0]
            arr = w.numpy()
            weight_specs.append({"name": w_name, "shape": list(arr.shape), "dtype": "float32"})
            bin_bytes.extend(arr.astype("<f4").tobytes())

    shard_name = "group1-shard1of1.bin"
    weights_manifest.append({"paths": [shard_name], "weights": weight_specs})

    with open(f"output/ranknet/{shard_name}", "wb") as f:
        f.write(bin_bytes)

    keras_ver = getattr(tf.keras, "__version__", "2.15.0")
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_ver}",
        "convertedBy": "TensorFlow.js Exporter",
        "modelTopology": {
            "keras_version": keras_ver,
            "backend": "tensorflow",
            "model_config": {
                "class_name": model.__class__.__name__,
                "config": model.get_config()
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
    }
}

with open("output/ranknet/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → output/ranknet/")
print(f"Saved → output/ranknet/meta.json")
