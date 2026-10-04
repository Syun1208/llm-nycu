from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.classes.settings import SimilarityRunConfig

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
PROBLEM1_DATA = DATA_DIR / "problem1"
PROBLEM2_DATA = DATA_DIR / "problem2"
APP_NAME = "llm_hw1"
DEFAULT_DEVICE_ID = 0
EXIT_OK, EXIT_RUNTIME_ERROR, EXIT_USAGE_ERROR, EXIT_INTERRUPTED = 0, 1, 2, 130

logger = logging.getLogger(APP_NAME)


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {path}")
    return path


def default_output_dir(*, name: str) -> Path:
    return OUTPUTS_DIR / f"{date.today():%Y-%m-%d}_{name}"


def run_problem1(args: argparse.Namespace) -> int:
    from src.data.classes.settings import CharLMSettings, LanguageIdRunConfig
    from src.services.language_id import run_language_id

    config = LanguageIdRunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir or default_output_dir(name="p1_language_id"),
        device_id=args.device_id,
        seed=args.seed,
        cells=tuple(args.cells),
        char_lm=CharLMSettings(epochs=args.epochs),
    )
    summary = run_language_id(config=config)
    for name, metrics in summary.items():
        print(f"{name}: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    return EXIT_OK


def build_similarity_config(
    *,
    args: argparse.Namespace,
    name: str,
) -> SimilarityRunConfig:
    from src.data.classes.settings import SimilarityRunConfig, SimilaritySettings
    from src.services.similarity import PRESETS

    return SimilarityRunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir or default_output_dir(name=name),
        device_id=args.device_id,
        members=PRESETS[args.preset],
        seed=args.seed,
        settings=SimilaritySettings(epochs=args.epochs),
    )


def run_problem2(args: argparse.Namespace) -> int:
    from src.services.similarity import run_similarity

    summary = run_similarity(
        config=build_similarity_config(
            args=args,
            name=f"p2_{args.preset}",
        )
    )
    print(", ".join(f"{key}={value:.4f}" for key, value in summary.items()))
    return EXIT_OK


def run_problem2_heads(args: argparse.Namespace) -> int:
    from src.services.similarity import run_head_comparison

    table = run_head_comparison(
        config=build_similarity_config(
            args=args,
            name="p2_head_comparison",
        )
    )
    print(table.to_string())
    return EXIT_OK


def add_common_arguments(
    *,
    parser: argparse.ArgumentParser,
    data_dir: Path,
    epochs: int,
) -> None:
    parser.add_argument(
        "--data-dir",
        type=existing_dir,
        default=data_dir,
        help="folder with train/val/test csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: outputs/<date>_<task>",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=DEFAULT_DEVICE_ID,
        help="CUDA device index (PCI order)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help="training epochs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show DEBUG logs on the console",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NYCU LLM homework 1 experiments",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    problem1 = subparsers.add_parser(
        "p1",
        help="Problem 1: n-gram, RNN and LSTM language identification",
    )
    add_common_arguments(
        parser=problem1,
        data_dir=PROBLEM1_DATA,
        epochs=40,
    )
    problem1.add_argument(
        "--cells",
        nargs="+",
        choices=("rnn", "lstm"),
        default=("rnn", "lstm"),
        help="neural models",
    )
    problem1.set_defaults(handler=run_problem1)

    problem2 = subparsers.add_parser(
        "p2",
        help="Problem 2: train the Transformer ensemble and write submission.csv",
    )
    add_common_arguments(
        parser=problem2,
        data_dir=PROBLEM2_DATA,
        epochs=20,
    )
    problem2.add_argument(
        "--preset",
        choices=("single", "v9"),
        default="v9",
        help="ensemble preset",
    )
    problem2.set_defaults(handler=run_problem2)

    heads = subparsers.add_parser(
        "p2-heads",
        help="Problem 2 (c): compare 1, 4 and 8 attention heads",
    )
    add_common_arguments(
        parser=heads,
        data_dir=PROBLEM2_DATA,
        epochs=20,
    )
    heads.set_defaults(
        handler=run_problem2_heads,
        preset="single",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src.utils.env import load_env
    from src.utils.log_setup import setup_logging

    log_file = setup_logging(
        app_name=APP_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger.info("Log file: %s", log_file)
    env_file = load_env(start=ROOT)
    logger.debug("Environment file: %s", env_file)
    logger.debug("Arguments: %s", vars(args))

    try:
        return args.handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        logger.debug(
            "Traceback",
            exc_info=True,
        )
        return EXIT_USAGE_ERROR
    except Exception:
        logger.exception("Run failed")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
