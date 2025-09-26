import argparse
import os
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare submission folder")
    parser.add_argument("--predictions", type=str, required=True, help="Directory containing predicted masks")
    parser.add_argument("--solution", type=str, required=True, help="Directory containing solution files (ckpt, requirements, run)")
    parser.add_argument("--output", type=str, required=True, help="Output directory to build submission structure")
    return parser.parse_args()


def validate_solution_dir(solution_dir: Path) -> None:
    required_files = {"ckpt.pth", "requirements.txt", "run.py"}
    existing = {path.name for path in solution_dir.iterdir() if path.is_file()}
    missing = required_files - existing
    if missing:
        raise FileNotFoundError(f"Missing required files in solution directory: {', '.join(sorted(missing))}")


def copy_files(file_paths: List[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in file_paths:
        dst = destination_dir / src.name
        dst.write_bytes(src.read_bytes())


def prepare_submission(predictions_dir: Path, solution_dir: Path, output_dir: Path) -> None:
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    if not solution_dir.is_dir():
        raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

    validate_solution_dir(solution_dir)

    masks_output = output_dir / "masks"
    masks_output.mkdir(parents=True, exist_ok=True)
    for mask_path in predictions_dir.glob("*.png"):
        target_path = masks_output / mask_path.name
        target_path.write_bytes(mask_path.read_bytes())

    solution_output = output_dir / "solution"
    solution_output.mkdir(parents=True, exist_ok=True)
    for required_name in ["ckpt.pth", "requirements.txt", "run.py"]:
        src = solution_dir / required_name
        dst = solution_output / required_name
        dst.write_bytes(src.read_bytes())


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions)
    solution_dir = Path(args.solution)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_submission(predictions_dir, solution_dir, output_dir)


if __name__ == "__main__":
    main()

