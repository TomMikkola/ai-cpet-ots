"""Validate publication files without executing private-data notebook workflows.

Run from any directory with the project development dependencies installed.
The checker is read-only: it never clears outputs, stages files, or modifies Git.
"""

import ast
import hashlib
import re
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEFERRED = "src/Pre-process-BBB-ECG.ipynb"
EXPECTED_NOTEBOOKS = {f"model-{name}.ipynb" for name in ("1", "2", "3", "4", "1-2", "1-3-4")} | {
    "statistics.ipynb",
    "preprocess.ipynb",
}
EXPECTED_FILES = (
    {
        ".gitattributes",
        ".github/workflows/checks.yml",
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "environments/README.md",
        "environments/preprocess-legacy.in",
        "environments/preprocess-legacy.txt",
        "scripts/check_publication.py",
        "models/checksums.sha256",
    }
    | {
        f"src/ai_cpet_ots/{name}.py"
        for name in (
            "__init__",
            "dataloaders",
            "helpers",
            "models",
            "optimizers",
            "paths",
            "testers",
        )
    }
    | {f"models/Model_{name}.pth" for name in ("1", "2", "3", "4", "1_2", "1_3_4")}
    | {
        f"notebooks/{'statistics/' if name == 'statistics.ipynb' else ''}{name}"
        for name in EXPECTED_NOTEBOOKS
    }
)


def check_source(source: str, label: str) -> None:
    """Reject machine-specific executable paths, stale imports, and wildcards."""
    tree = ast.parse(source, filename=label)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names), f"Wildcard import: {label}"
            assert node.module not in {
                "src",
                "global_vars",
                "models",
                "dataloaders",
                "helpers",
                "testers",
                "optimizers",
            }, f"Legacy import: {label}"
            assert not (node.module or "").startswith("src."), f"Legacy src import: {label}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.match(r"(?:[A-Za-z]:[\\/]|/(?:Users|home)/)", node.value), (
                f"Machine-specific path: {label}"
            )
            assert "../Models" not in node.value and "../models" not in node.value, (
                f"Working-directory path: {label}"
            )


def check_notebooks() -> int:
    """Validate schema, cleared outputs, portable code, and non-training defaults."""
    paths = list((ROOT / "notebooks").rglob("*.ipynb"))
    assert (
        len(paths) == len(EXPECTED_NOTEBOOKS)
        and {path.name for path in paths} == EXPECTED_NOTEBOOKS
    )
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert set(notebook.metadata) <= {"kernelspec", "language_info"}, (
            f"Local notebook metadata: {path.name}"
        )
        source = []
        for cell in notebook.cells:
            assert not cell.get("attachments"), f"Embedded attachment: {path.name}"
            if cell.cell_type == "code":
                assert not cell.outputs, f"Saved outputs: {path.name}"
                assert cell.execution_count is None, f"Execution count: {path.name}"
                source.append(cell.source)
        combined = "\n".join(source)
        check_source(combined, path.name)
        if path.name == "preprocess.ipynb":
            # Public CI must validate legacy syntax without importing its SDK.
            ast.parse(combined, filename=path.name, feature_version=(3, 7))
            assert not any(
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("ai_cpet_ots")
                or isinstance(node, ast.Import)
                and any(alias.name.startswith("ai_cpet_ots") for alias in node.names)
                for node in ast.walk(ast.parse(combined))
            ), "Legacy preprocessing must not import the Python 3.11 package"
        if path.name.startswith("model-") or path.name == "preprocess.ipynb":
            tree = ast.parse(combined)
            defaults = (
                ("run_preprocessing", "plot_diagnostics")
                if path.name == "preprocess.ipynb"
                else ("training_model", "save_model")
            )
            for name in defaults:
                values = [
                    node.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in node.targets
                    )
                ]
                assert (
                    len(values) == 1
                    and isinstance(values[0], ast.Constant)
                    and values[0].value is False
                ), f"Unsafe default {name}: {path.name}"
    return len(paths)


def check_git_files() -> int:
    """Verify exclusions and inspect the candidate publication file inventory."""
    excluded = [
        DEFERRED,
        "data/private.xlsx",
        "outputs/predictions.xlsx",
        "AI-CPET-OTS/pyvenv.cfg",
        "src/__pycache__/module.pyc",
        ".publication-baseline/hashes.json",
        ".cache/example",
        ".venv/pyvenv.cfg",
        "recording.npy",
        "recording.adicht",
        ".venv-preprocess/pyvenv.cfg",
        "tests/test_paths.py",
        "VALIDATION.md",
    ]
    for name in excluded:
        result = subprocess.run(["git", "check-ignore", "--quiet", name], cwd=ROOT, check=False)
        assert result.returncode == 0, f"Missing Git exclusion: {name}"
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = [name for name in files if name]
    assert set(files) == EXPECTED_FILES, (
        f"Unexpected publication files: {sorted(set(files) - EXPECTED_FILES)}; "
        f"missing required files: {sorted(EXPECTED_FILES - set(files))}"
    )
    forbidden = {".xlsx", ".xls", ".csv", ".tsv", ".npy", ".npz", ".adicht", ".pyc", ".bkp"}
    for name in files:
        assert Path(name).suffix.lower() not in forbidden, (
            f"Private or generated publication file: {name}"
        )
        assert name not in excluded, f"Excluded file already tracked: {name}"
        assert not any(
            part in {"data", "outputs", "__pycache__", ".venv", ".cache", ".publication-baseline"}
            for part in Path(name).parts
        ), f"Local publication file: {name}"
    for path in (ROOT / "src/ai_cpet_ots").glob("*.py"):
        check_source(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    manifest = (ROOT / "models/checksums.sha256").read_text().splitlines()
    assert len(manifest) == 6
    for line in manifest:
        digest, filename = line.split()
        assert hashlib.sha256((ROOT / "models" / filename).read_bytes()).hexdigest() == digest, (
            f"Changed checkpoint: {filename}"
        )
    assert len(list((ROOT / "models").glob("*.pth"))) == 6
    return len(files)


def main() -> None:
    """Report notebook and Git inventory validation, raising on the first failure."""
    count = check_notebooks()
    files = check_git_files()
    print(
        f"Publication checks passed: {count} clean notebooks; {files} candidate files; six original checkpoints."
    )


if __name__ == "__main__":
    main()
