"""
convert_csv_to_standard.py
==========================
Converts 'master_leaderboard_rows.csv' into the standard exam JSON format:
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

Outputs to data/converted_master_leaderboard.json
"""

import os
import csv
import json
from collections import defaultdict


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        s = str(val).strip()
        if not s:
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=None):
    if val is None:
        return default
    try:
        s = str(val).strip()
        if not s:
            return default
        return int(float(s))
    except (ValueError, TypeError):
        return default


def convert_csv(csv_path, output_path):
    print(f"Reading CSV from: {csv_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows read from CSV: {len(rows)}")

    # First pass: compute test-level statistics and collect ranks/scores
    test_scores = defaultdict(list)
    test_subject_avgs = defaultdict(lambda: defaultdict(list))
    test_subject_toppers = defaultdict(lambda: defaultdict(list))
    test_max_marks = defaultdict(list)
    test_ranks = defaultdict(list)
    test_rows = defaultdict(list)

    for idx, r in enumerate(rows):
        tname = (r.get("test_name") or "").strip()
        if not tname:
            continue

        score = safe_float(r.get("score"))
        max_m = safe_float(r.get("max_marks"))
        rank = safe_int(r.get("rank"))
        pct = safe_float(r.get("percentile"))

        if max_m > 0:
            test_max_marks[tname].append(max_m)
        if score > 0:
            test_scores[tname].append(score)
        if rank and rank > 0:
            test_ranks[tname].append(rank)

        test_rows[tname].append((idx, r))

        subj_raw = r.get("subject_data")
        if subj_raw:
            try:
                subjs = json.loads(subj_raw)
                for s in subjs:
                    sname = (s.get("subjectName") or "").strip().lower()
                    if sname:
                        avg_m = safe_float(s.get("totalAvgMarks"))
                        high_m = safe_float(s.get("highestMarks"))
                        if avg_m > 0:
                            test_subject_avgs[tname][sname].append(avg_m)
                        if high_m > 0:
                            test_subject_toppers[tname][sname].append(high_m)
            except Exception:
                pass

    # Aggregate test-level metadata
    test_meta = {}
    for tname in set(list(test_scores.keys()) + list(test_max_marks.keys())):
        scores = test_scores[tname]
        max_marks_list = test_max_marks[tname]
        ranks_list = test_ranks[tname]
        max_m = max(max_marks_list) if max_marks_list else 300.0

        if scores:
            topper_val = max(scores)
        else:
            topper_val = sum(
                max(v) for v in test_subject_toppers[tname].values()
            ) if test_subject_toppers[tname] else max_m

        subj_avgs = test_subject_avgs[tname]
        if subj_avgs:
            avg_val = sum(sum(v) / len(v) for v in subj_avgs.values())
        elif scores:
            avg_val = sum(scores) / len(scores)
        else:
            avg_val = max_m * 0.4

        # Total student count estimate for this test
        max_rank = max(ranks_list) if ranks_list else 0
        total_in_test = max(max_rank, len(test_rows[tname]), 50)

        test_meta[tname] = {
            "maxMarks": max_m,
            "topper": round(topper_val, 2),
            "avg": round(avg_val, 2),
            "totalStudents": total_in_test,
        }

    # Second pass: transform rows into standard format
    standard_records = []
    skipped = 0

    for tname, items in test_rows.items():
        meta = test_meta.get(tname, {"maxMarks": 300.0, "topper": 300.0, "avg": 120.0, "totalStudents": 100})
        total_students = meta["totalStudents"]

        # Sort items by score descending to assign synthetic rank if rank is missing
        sorted_by_score = sorted(items, key=lambda x: safe_float(x[1].get("score")), reverse=True)
        score_rank_map = {}
        curr_rank = 1
        for i, (orig_idx, row_data) in enumerate(sorted_by_score):
            score_rank_map[orig_idx] = curr_rank
            curr_rank += 1

        for orig_idx, r in items:
            score = safe_float(r.get("score"))
            max_marks = safe_float(r.get("max_marks"))
            percentile = safe_float(r.get("percentile"))
            rank = safe_int(r.get("rank"))
            exam_date = (r.get("exam_date") or "").strip()

            if max_marks <= 0:
                max_marks = meta["maxMarks"]

            avg = meta["avg"]
            topper = max(meta["topper"], score)

            # Deduce or correct rank/percentile if missing or 0
            if rank is None or rank <= 0:
                rank = score_rank_map.get(orig_idx, 1)

            if percentile <= 0 and score > 0:
                # Calculate percentile from rank and totalStudents
                percentile = max(0.01, min(99.99, (1.0 - float(rank) / float(total_students)) * 100.0))

            # Parse subject data
            subjects = {"physics": None, "chemistry": None, "maths": None}
            subj_raw = r.get("subject_data")
            if subj_raw:
                try:
                    subjs = json.loads(subj_raw)
                    for s in subjs:
                        sname = (s.get("subjectName") or "").strip().lower()
                        target_key = None
                        if "phys" in sname:
                            target_key = "physics"
                        elif "chem" in sname:
                            target_key = "chemistry"
                        elif "math" in sname:
                            target_key = "maths"

                        if target_key:
                            s_pct = safe_float(s.get("percentile"))
                            if s_pct <= 0:
                                s_pct = percentile
                            subjects[target_key] = {
                                "score": safe_float(s.get("totalMarks")),
                                "percentile": round(s_pct, 2),
                                "avg": safe_float(s.get("totalAvgMarks")),
                                "topper": safe_float(s.get("highestMarks")),
                            }
                except Exception:
                    pass

            rec = {
                "testName": tname,
                "examDate": exam_date,
                "score": score,
                "maxMarks": max_marks,
                "avg": avg,
                "topper": topper,
                "rank": rank,
                "percentile": round(percentile, 2),
                "physics": subjects["physics"] or {"score": 0, "percentile": 0, "avg": 0, "topper": 0},
                "chemistry": subjects["chemistry"] or {"score": 0, "percentile": 0, "avg": 0, "topper": 0},
                "maths": subjects["maths"] or {"score": 0, "percentile": 0, "avg": 0, "topper": 0},
            }
            standard_records.append(rec)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standard_records, f, indent=2)

    print(f"Successfully converted {len(standard_records)} records (skipped {skipped}).")
    print(f"Saved to: {output_path}")
    return len(standard_records)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "master_leaderboard_rows.csv")
    out_file = os.path.join(base_dir, "data", "converted_master_leaderboard.json")
    convert_csv(csv_file, out_file)
