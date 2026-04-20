"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton, CUDA, or Python) into a Solution JSON file for submission. When the
repository uses the multi-definition layout recommended by the FAQ, run this
script from the repository root to pack every top-level directory containing a
config.toml, or pass one or more directories explicitly.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config(project_dir: Path) -> dict:
    """Load configuration from config.toml."""
    config_path = project_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_source_dir(project_dir: Path, build_config: dict) -> Path:
    """Return the source directory for a solution project."""
    if "source_dir" in build_config:
        return project_dir / "solution" / build_config["source_dir"]

    language = build_config["language"]
    if language == "triton":
        return project_dir / "solution" / "triton"
    if language == "cuda":
        return project_dir / "solution" / "cuda"
    if language == "python":
        return project_dir / "solution" / "python"
    raise ValueError(f"Unsupported language: {language}")


def pack_solution(project_dir: Path, output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    project_dir = project_dir.resolve()
    config = load_config(project_dir)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]
    source_dir = get_source_dir(project_dir, build_config)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = project_dir / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def discover_solution_dirs(paths: list[str]) -> list[Path]:
    """Find solution project directories to pack."""
    if paths:
        return [Path(path) for path in paths]

    if (PROJECT_ROOT / "config.toml").exists():
        return [PROJECT_ROOT]

    solution_dirs = sorted(
        path for path in PROJECT_ROOT.iterdir()
        if path.is_dir() and (path / "config.toml").exists()
    )
    if not solution_dirs:
        raise FileNotFoundError(
            f"No config.toml found in {PROJECT_ROOT} or its top-level subdirectories"
        )
    return solution_dirs


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Solution directories to pack (default: root config or all top-level configs)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (only valid with one solution directory)",
    )
    args = parser.parse_args()

    try:
        solution_dirs = discover_solution_dirs(args.paths)
        if args.output is not None and len(solution_dirs) != 1:
            raise ValueError("--output can only be used when packing one solution directory")
        for solution_dir in solution_dirs:
            pack_solution(solution_dir, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
