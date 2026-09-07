"""
convert_all.py
==============
Unified runner to convert non-standard files (master_leaderboard_rows.csv and lb.json)
into the standard format JSON files in data/ directory, and summarize all available training data.
"""

import os
import glob
import json
from convert_csv_to_standard import convert_csv
from convert_lb_to_standard import convert_lb


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    csv_file = os.path.join(base_dir, "master_leaderboard_rows.csv")
    csv_out = os.path.join(data_dir, "converted_master_leaderboard.json")

    lb_file = os.path.join(data_dir, "lb.json")
    lb_out = os.path.join(data_dir, "converted_lb.json")

    print("=" * 60)
    print("STEP 1: Converting master_leaderboard_rows.csv ...")
    print("=" * 60)
    n_csv = convert_csv(csv_file, csv_out)

    print("\n" + "=" * 60)
    print("STEP 2: Converting lb.json ...")
    print("=" * 60)
    n_lb = convert_lb(lb_file, lb_out, data_dir=data_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Dataset Summary in data/ ...")
    print("=" * 60)
    all_json_files = glob.glob(os.path.join(data_dir, "*.json"))
    total_records = 0
    file_counts = {}

    for fpath in all_json_files:
        fname = os.path.basename(fpath)
        if fname == "lb.json":
            continue  # lb.json is the raw unconverted leaderboard
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list):
                cnt = len(d)
                file_counts[fname] = cnt
                total_records += cnt
        except Exception as e:
            file_counts[fname] = f"Error: {e}"

    for fn, count in sorted(file_counts.items(), key=lambda x: str(x[0])):
        print(f"  - {fn:<36}: {count} records")

    print("-" * 60)
    print(f"Total standard records available for training: {total_records}")
    print("All conversions complete!")


if __name__ == "__main__":
    main()
