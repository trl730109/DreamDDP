#!/usr/bin/env python3
"""
Calculate the speedup of DreamDDP relative to pipe_sgd and localsgd from cpu_clock results and write to speedup.txt.
Formula: (other algorithm time - dreamddp time) / dreamddp time
"""
import argparse
import os


def get_elapsed(filepath: str):
    if not filepath or not os.path.isfile(filepath):
        return None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("elapsed:"):
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser(description="DreamDDP speedup stats from cpu_clock")
    parser.add_argument("--time_stamp", type=str, required=True, help="Experiment time_stamp (e.g. gpt2-20250308_123456)")
    parser.add_argument("--dnn", type=str, required=True)
    parser.add_argument("--nworkers", type=int, required=True)
    parser.add_argument("--bandwidth", type=str, required=True)
    args = parser.parse_args()

    # use the results directory in the root of the repository
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_base = os.path.join(repo_root, "results", args.time_stamp)
    subdir = f"{args.dnn}-{args.nworkers}-{args.bandwidth}"

    dreamddp_path = os.path.join(results_base, "cpu_clocks", "dreamddp", subdir, "cpu_clock_time_rank0.txt")
    dreamddp_t = get_elapsed(dreamddp_path)

    lines = []
    lines.append("========== DreamDDP speedup (other - dreamddp) / dreamddp ==========")

    if dreamddp_t is None or dreamddp_t == 0:
        lines.append("No cpu_clock data (set cpu_clock=True to get)")
    else:
        for alg_name, label in [("pipe_sgd", "s1"), ("localsgd", "s2")]:
            path = os.path.join(results_base, "cpu_clocks", alg_name, subdir, "cpu_clock_time_rank0.txt")
            other_t = get_elapsed(path)
            if other_t is not None:
                ratio = (other_t - dreamddp_t) / dreamddp_t
                lines.append(f"  vs {alg_name} ({label}): {1+ratio:.4f}x")
            else:
                lines.append(f"  vs {alg_name} ({label}): (No cpu_clock data, set cpu_clock=True to get)")

    out_dir = results_base
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speedup.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[speedup stats saved to {out_path}]")


if __name__ == "__main__":
    main()
