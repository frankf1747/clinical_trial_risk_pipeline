import zipfile

import pytest

from ctrisk.ingest.aact import TABLES, fetch


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(f"export/{name}", "nct_id\nNCT001\n")


def test_extracts_only_needed_tables(tmp_path):
    zip_path = tmp_path / "aact.zip"
    make_zip(zip_path, [f"{t}.txt" for t in TABLES] + ["outcomes.txt"])

    out = fetch(str(zip_path), tmp_path / "aact")

    assert sorted(p.name for p in out) == sorted(f"{t}.txt" for t in TABLES)
    assert not (tmp_path / "aact" / "outcomes.txt").exists()


def test_missing_table_fails_loudly(tmp_path):
    zip_path = tmp_path / "aact.zip"
    make_zip(zip_path, ["studies.txt"])

    with pytest.raises(FileNotFoundError, match="designs.txt"):
        fetch(str(zip_path), tmp_path / "aact")
