"""experiments/run.py — P2 Part C CLI entrypoint.

Thin wrapper over model/main.py: resolves --teacher to the locked
p3_reach_{baseline,spiking}.py config (see P2_checklist.md Part A/B),
maps --seed/--grid to the P12_SEED/GRID env vars those two configs read,
and forwards everything else to model/main.py's own CLI unchanged.

Example (one run):
    python experiments/run.py --teacher python  --seed 0 --grid 7x7
    python experiments/run.py --teacher spiking --seed 0 --grid 7x7

Example (the standard 9-seed P_3 sweep, both conditions):
    python experiments/run_seed_sweep.py --teacher python
    python experiments/run_seed_sweep.py --teacher spiking
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "python":  REPO_ROOT / "experiments/configs/p3_reach_baseline.py",
    "spiking": REPO_ROOT / "experiments/configs/p3_reach_spiking.py",
    # burst: the burst-plasticity hidden-layer
    # actor variant -- experimental, not one of the two P2-locked
    # conditions, kept here (not a separate --config override) so the
    # quick-gate/9-seed sweep tooling stays uniform.
    "burst":   REPO_ROOT / "experiments/configs/p3_reach_burst.py",
    # combined: joints/object encoder swap only (Grid2DPlaceFieldEncoder +
    # LinearRateEncoder), see p3_reach_combined_encoder.py's own docstring.
    "combined": REPO_ROOT / "experiments/configs/p3_reach_combined_encoder.py",
    # bypass: D1/D2
    # read the combined encoder directly, skipping the recurrent cortex
    # (which was measured to collapse the signal) -- see
    # p3_reach_bypass_cortex.py's own docstring.
    "bypass": REPO_ROOT / "experiments/configs/p3_reach_bypass_cortex.py",
    # fixedhidden: place-field encoder -> frozen random hidden layer -> D1/D2, cortex
    # bypassed, readout trained, efference on -- see
    # p3_reach_fixedhidden.py's own docstring.
    "fixedhidden": REPO_ROOT / "experiments/configs/p3_reach_fixedhidden.py",
}
GRID_CHOICES = ["6x6", "7x7", "8x8", "10x10", "12x12", "16x16"]
# Only one critic architecture is locked (the P1.4 signed-weight single-
# pool readout, see findings/P14_OPPONENT_POOL_CANCELLATION_FINDING.md) --
# this flag exists for forward compatibility, not because there's a real
# choice today.
CRITIC_MODES = ["signedweight"]


def main():
    parser = argparse.ArgumentParser(description="P3 reach-task runner (P2-locked config)")
    parser.add_argument("--teacher", choices=sorted(CONFIGS), required=True,
                        help="python: actor trained from the analytic TD critic (baseline). "
                             "spiking: actor trained from the spiking critic's own delta "
                             "(P1.4 signed-weight readout).")
    parser.add_argument("--critic-mode", choices=CRITIC_MODES, default="signedweight")
    parser.add_argument("--seed", type=int, required=True,
                        help="0-8 for the standard 9-seed sweep; sets P12_SEED (the config "
                             "applies its own seed+1 NEST-seed offset internally)")
    parser.add_argument("--grid", type=str, default="7x7", choices=GRID_CHOICES)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--restore-checkpoint", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["P12_SEED"] = str(args.seed)
    env["GRID"] = args.grid

    cmd = [sys.executable, str(REPO_ROOT / "model/main.py"),
           "--config", str(CONFIGS[args.teacher])]
    # --seed is deliberately NOT forwarded to model/main.py's own --seed --
    # that would overwrite the seed+1 NEST offset the config computes from
    # P12_SEED, the convention every P1.x/P2 run used.
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    if args.out_dir is not None:
        cmd += ["--out-dir", args.out_dir]
    if args.restore_checkpoint is not None:
        cmd += ["--restore-checkpoint", args.restore_checkpoint]
    if args.verbose:
        cmd += ["-v"]

    print(f"[run.py] teacher={args.teacher} critic_mode={args.critic_mode} "
          f"seed={args.seed} grid={args.grid}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
