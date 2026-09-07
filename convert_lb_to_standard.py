"""
convert_lb_to_standard.py
=========================
Converts 'data/lb.json' into the standard exam JSON format:
[
  {
    "testName": "...",
    "examDate": "...",
    "score": ...,
    "maxMarks": ...,
    "avg": ...,
    "topper": ...,
    "rank": ...,
    "percentile": ...,
    "physics": {"score": ..., "percentile": ..., "avg": ..., "topper": ...},
    "chemistry": {"score": ..., "percentile": ..., "avg": ..., "topper": ...},
    "maths": {"score": ..., "percentile": ..., "avg": ..., "topper": ...}
  },
  ...
]

Outputs to data/converted_lb.json
"""

import os
import glob
import json
import numpy as np


def build_test_metadata_index(data_dir):
    """
    Scans standard JSON files in data/ to build a metadata lookup table
    for tests (maxMarks, avg, topper, totalStudents, subject stats).
    """
    test_meta = {}
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for fpath in json_files:
        fname = os.path.basename(fpath)
        if fname in ("lb.json", "converted_lb.json"):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for r in data:
                tname = (r.get("testName") or "").strip()
                if not tname:
                    continue

                max_m = r.get("maxMarks")
                avg = r.get("avg")
                top = r.get("topper")
                rank = r.get("rank")
                pct = r.get("percentile")

                if tname not in test_meta:
                    test_meta[tname] = {
                        "maxMarks": [],
                        "avg": [],
                        "topper": [],
                        "totalStudents": [],
                        "physics": {"avg": [], "topper": []},
                        "chemistry": {"avg": [], "topper": []},
                        "maths": {"avg": [], "topper": []},
                    }

                if max_m and max_m > 0:
                    test_meta[tname]["maxMarks"].append(float(max_m))
                if avg and avg > 0:
                    test_meta[tname]["avg"].append(float(avg))
                if top and top > 0:
                    test_meta[tname]["topper"].append(float(top))

                # Estimate total students from rank & percentile
                if rank and pct and 0 < pct < 100 and rank > 0:
                    try:
                        tot = float(rank) / (1.0 - float(pct) / 100.0)
                        if tot >= rank:
                            test_meta[tname]["totalStudents"].append(tot)
                    except ZeroDivisionError:
                        pass

                for subj_key in ("physics", "chemistry", "maths"):
                    subj = r.get(subj_key)
                    if isinstance(subj, dict):
                        s_avg = subj.get("avg")
                        s_top = subj.get("topper")
                        if s_avg and s_avg > 0:
                            test_meta[tname][subj_key]["avg"].append(float(s_avg))
                        if s_top and s_top > 0:
                            test_meta[tname][subj_key]["topper"].append(float(s_top))

        except Exception as e:
            print(f"  [WARN] Skipping {fname} during metadata indexing: {e}")

    # Reduce lists to median values
    aggregated = {}
    for tname, vals in test_meta.items():
        max_m = float(np.median(vals["maxMarks"])) if vals["maxMarks"] else 300.0
        avg_val = float(np.median(vals["avg"])) if vals["avg"] else max_m * 0.4
        top_val = float(np.median(vals["topper"])) if vals["topper"] else max_m
        tot_students = float(np.median(vals["totalStudents"])) if vals["totalStudents"] else 500.0

        subj_stats = {}
        for subj_key in ("physics", "chemistry", "maths"):
            s_avg = float(np.median(vals[subj_key]["avg"])) if vals[subj_key]["avg"] else avg_val / 3.0
            s_top = float(np.median(vals[subj_key]["topper"])) if vals[subj_key]["topper"] else max_m / 3.0
            subj_stats[subj_key] = {"avg": s_avg, "topper": s_top}

        aggregated[tname] = {
            "maxMarks": max_m,
            "avg": avg_val,
            "topper": top_val,
            "totalStudents": tot_students,
            "subjects": subj_stats,
        }

    return aggregated


def convert_lb(lb_path, output_path, data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(lb_path)

    print(f"Reading lb.json from: {lb_path}")
    with open(lb_path, "r", encoding="utf-8") as f:
        lb_data = json.load(f)

    meta_index = build_test_metadata_index(data_dir)
    print(f"Indexed {len(meta_index)} test profiles from existing data files.")

    standard_records = []
    tests = lb_data.get("tests", [])
    print(f"Found {len(tests)} test suites in lb.json.")

    for test_obj in tests:
        tname = (test_obj.get("testName") or "").strip()
        summary = test_obj.get("summary", {})
        exam_date = summary.get("examDate") or ""
        lb_entries = summary.get("leaderboard", [])

        if not lb_entries:
            continue

        # Look up test metadata or fuzzy match
        meta = meta_index.get(tname)
        if not meta:
            # Try case/whitespace insensitive match
            for k, v in meta_index.items():
                if k.strip().lower() == tname.strip().lower():
                    meta = v
                    break

        # Fallback if metadata not found
        if not meta:
            scores = [e.get("totalMarks", 0) for e in lb_entries if e.get("totalMarks")]
            max_achieved = max(scores) if scores else 300.0
            if max_achieved > 200:
                max_marks = 300.0
            elif max_achieved > 100:
                max_marks = 198.0
            else:
                max_marks = 100.0

            meta = {
                "maxMarks": max_marks,
                "avg": round(max_marks * 0.4, 2),
                "topper": max_achieved,
                "totalStudents": 500.0,
                "subjects": {
                    "physics": {"avg": max_marks * 0.13, "topper": max_marks / 3.0},
                    "chemistry": {"avg": max_marks * 0.13, "topper": max_marks / 3.0},
                    "maths": {"avg": max_marks * 0.13, "topper": max_marks / 3.0},
                }
            }

        max_marks = meta["maxMarks"]
        avg = meta["avg"]
        topper = meta["topper"]
        total_students = meta["totalStudents"]
        subj_stats = meta["subjects"]

        for entry in lb_entries:
            rank = entry.get("ranks")
            score = entry.get("totalMarks")

            if rank is None or rank <= 0 or score is None:
                continue

            # Ensure total_students is >= rank
            if total_students < rank:
                total_students = float(rank) / 0.1  # assume top 10%

            percentile = max(0.0, min(100.0, (1.0 - float(rank) / float(total_students)) * 100.0))

            # Parse subject performance
            # Each entry has subjectPerformance: [{subjectId, totalMarks}, ...]
            # Standard order is Physics (0), Chemistry (1), Maths (2)
            subj_perf = entry.get("subjectPerformance", [])
            subj_names = ["physics", "chemistry", "maths"]
            subjects = {}

            for idx, sname in enumerate(subj_names):
                s_score = 0.0
                if idx < len(subj_perf):
                    s_score = float(subj_perf[idx].get("totalMarks") or 0.0)

                s_avg = subj_stats[sname]["avg"]
                s_top = subj_stats[sname]["topper"]
                s_pct = max(0.0, min(100.0, percentile))

                subjects[sname] = {
                    "score": s_score,
                    "percentile": round(s_pct, 2),
                    "avg": round(s_avg, 2),
                    "topper": round(s_top, 2),
                }

            rec = {
                "testName": tname,
                "examDate": exam_date,
                "score": float(score),
                "maxMarks": float(max_marks),
                "avg": float(avg),
                "topper": float(topper),
                "rank": int(rank),
                "percentile": round(percentile, 2),
                "physics": subjects["physics"],
                "chemistry": subjects["chemistry"],
                "maths": subjects["maths"],
            }
            standard_records.append(rec)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standard_records, f, indent=2)

    print(f"Successfully converted {len(standard_records)} leaderboard records.")
    print(f"Saved to: {output_path}")
    return len(standard_records)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lb_file = os.path.join(base_dir, "data", "lb.json")
    out_file = os.path.join(base_dir, "data", "converted_lb.json")
    convert_lb(lb_file, out_file)
