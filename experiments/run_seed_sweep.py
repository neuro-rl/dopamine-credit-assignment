"""experiments/run_seed_sweep.py — P2 Part C seed-loop wrapper. Runs
seeds 0-8 for one --teacher condition through experiments/run.py, one
command per P_3 table/figure.

Example:
    python experiments/run_seed_sweep.py --teacher python  --grid 7x7
    python experiments/run_seed_sweep.py --teacher spiking --grid 7x7
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run the standard 9-seed P_3 sweep for one condition")
    parser.add_argument("--teacher", choices=["python", "spiking", "burst", "combined", "bypass", "fixedhidden"], required=True)
    parser.add_argument("--critic-mode", choices=["signedweight"], default="signedweight")
    parser.add_argument("--grid", type=str, default="7x7")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=9, help="number of seeds, 0..seeds-1")
    args = parser.parse_args()

    failures = []
    for seed in range(args.seeds):
        cmd = [sys.executable, str(REPO_ROOT / "experiments/run.py"),
               "--teacher", args.teacher, "--critic-mode", args.critic_mode,
               "--seed", str(seed), "--grid", args.grid]
        if args.episodes is not None:
            cmd += ["--episodes", str(args.episodes)]
        print(f"\n=== seed {seed}/{args.seeds - 1} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(seed)
            print(f"[run_seed_sweep] seed {seed} FAILED (exit {result.returncode})")

    if failures:
        print(f"\n[run_seed_sweep] {len(failures)}/{args.seeds} seeds failed: {failures}")
        sys.exit(1)
    print(f"\n[run_seed_sweep] all {args.seeds} seeds completed.")


if __name__ == "__main__":
    main()
