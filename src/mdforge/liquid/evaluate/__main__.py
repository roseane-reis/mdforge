"""CLI: ``python -m mdforge.liquid.evaluate --config water.yaml``.

Also exposed as the ``mdforge-eval`` console script. Loads a config (or
auto-discovers legs from a campaign run directory), runs the evaluation, writes
the report, and prints the headline verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import EvalConfig
from .ingest import legs_from_campaign
from .pipeline import run_evaluation
from .report import build_evaluation_report, format_console_summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdforge-eval",
        description="Evaluate a water model against experiment (TIP3P quality bar).",
    )
    p.add_argument("--config", type=Path, help="YAML config file.")
    p.add_argument("--campaign", type=Path,
                   help="Campaign run directory: auto-discover legs (overrides config.legs). "
                        "Requires --config for model/system metadata unless a config.yaml "
                        "is present in the run dir.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (overrides config.output.dir).")
    p.add_argument("--baseline", default="tip3p", help="Baseline model for the bar.")
    p.add_argument("--no-state-guard", action="store_true",
                   help="Skip the 298.15 K / 1 atm state check (results not comparable).")
    p.add_argument("--no-plots", action="store_true", help="Do not render plots.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per trajectory (fast smoke test).")
    return p


def _load_config(args) -> EvalConfig:
    if args.config:
        config = EvalConfig.from_yaml(args.config)
    elif args.campaign and (args.campaign / "config.yaml").is_file():
        config = EvalConfig.from_yaml(args.campaign / "config.yaml")
    else:
        raise SystemExit("error: --config is required (no config.yaml in the campaign dir)")
    if args.campaign:
        config.legs = legs_from_campaign(args.campaign)
        config.base_dir = Path(".")  # discovered leg paths are absolute
        config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _load_config(args)

    out_dir = args.out_dir or config.resolve(config.output.dir)
    result = run_evaluation(config, enforce_state=not args.no_state_guard,
                            max_frames=args.max_frames)
    artifacts = build_evaluation_report(
        result, outdir=out_dir, baseline_model=args.baseline,
        make_plots=(not args.no_plots and config.output.plots),
    )
    print("\n" + format_console_summary(
        result, artifacts["rating"], artifacts["reference"], baseline_model=args.baseline))
    for w in result.warnings:
        print(f"  [warn] {w}", file=sys.stderr)
    if artifacts.get("report_md"):
        print(f"\nWrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
