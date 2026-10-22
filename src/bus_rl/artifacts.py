"""Export compact run evidence to Git without training checkpoints or diagnostic logs."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from bus_rl.provenance import canonical_hash, file_hash, require_fresh_output

EVIDENCE_SUFFIXES = {".json", ".csv", ".toml", ".md", ".png", ".svg", ".html"}
LOCAL_ONLY = {"validation_metrics.csv", "progress.json"}


def verify_artifacts(path: Path) -> dict:
    manifest = json.loads((path / "artifacts.json").read_text())
    expected = set(manifest["files"])
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if actual != expected | {"artifacts.json"}:
        raise ValueError("artifact inventory mismatch: unexpected or missing files")
    for name, item in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid artifact path")
        target = path / relative
        if (
            target.is_symlink()
            or target.stat().st_size != item["bytes"]
            or file_hash(target) != item["sha256"]
        ):
            raise ValueError(f"artifact checksum mismatch: {name}")
    if canonical_hash(manifest["files"]) != manifest["inventory_sha256"]:
        raise ValueError("artifact manifest checksum mismatch")
    return manifest


def export_artifacts(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    require_fresh_output(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("artifact source and output must be separate directory trees")
    status = source / "status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
        raise ValueError("only a completed experiment can be exported")
    selected, omitted = [], []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not supported in run evidence: {path}")
        if not path.is_file():
            continue
        if path.suffix in EVIDENCE_SUFFIXES and path.name not in LOCAL_ONLY:
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError(f"artifact exceeds 8 MiB; retain raw output locally: {path}")
            selected.append(path)
        else:
            omitted.append(
                {"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size}
            )
    if not (source / "protocol.json").is_file():
        raise ValueError("experiment protocol.json is missing")
    output.mkdir(parents=True)
    files = {}
    for path in selected:
        relative = path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative.as_posix()] = {"sha256": file_hash(target), "bytes": target.stat().st_size}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "purpose": "Reproducible evidence for human-authored final reports",
        "files": files,
        "inventory_sha256": canonical_hash(files),
        "local_only": omitted,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }
    (output / "artifacts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_artifacts(output)
    return manifest
