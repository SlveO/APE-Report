"""Restore the accepted follow-up package without replacing differing evidence."""
import json
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parent


def main():
    package_dir = REPO / "followup2604-20261117"
    archive = package_dir / "FOLLOWUP2604_ACCEPTED.zip"
    listing = json.loads((package_dir / "FILE_LIST.json").read_text(encoding="utf-8"))
    expected = {"FOLLOWUP2604_ACCEPTED/" + r["path"]: r["bytes"] for r in listing["files"]}
    dest = REPO / "restored/followup2604"
    logdir = OUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    record = {"archive": str(archive.relative_to(REPO)), "destination": str(dest.relative_to(REPO)),
              "expected_files": listing["file_count"], "restored": 0,
              "existing_identical": 0, "different_existing": [], "missing_or_size_mismatch": []}
    with zipfile.ZipFile(archive) as z:
        entries = {r.filename: r for r in z.infolist() if not r.is_dir()}
        assert set(entries) == set(expected), "Archive and FILE_LIST paths differ"
        for name, size in expected.items():
            if entries[name].file_size != size:
                record["missing_or_size_mismatch"].append(name)
        assert not record["missing_or_size_mismatch"], record
        for name in sorted(entries):
            path = dest / name
            assert path.resolve().is_relative_to(dest.resolve()), name
            data = z.read(entries[name])
            if path.exists():
                if path.read_bytes() == data:
                    record["existing_identical"] += 1
                else:
                    record["different_existing"].append(name)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            record["restored"] += 1
    record["status"] = "PRESERVED_DIFFERENCES" if record["different_existing"] else "PASS"
    (logdir / "package_restore.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    if record["different_existing"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
