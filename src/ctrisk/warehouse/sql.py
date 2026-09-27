"""Run SQL files on any DB-API connection: Snowflake in the pipeline, DuckDB in tests."""
import re
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


def statements(path: Path) -> list[str]:
    """Split on semicolons that end a line; drop chunks that are only comments.

    Convention: never end a line with ';' inside a comment or string literal.
    """
    chunks = re.split(r";\s*$", path.read_text(), flags=re.MULTILINE)
    return [c.strip() for c in chunks
            if any(line.strip() and not line.strip().startswith("--") for line in c.splitlines())]


def run_files(cursor, paths: list[Path]) -> None:
    for path in paths:
        for statement in statements(path):
            cursor.execute(statement)


def failed_checks(cursor, path: Path) -> list[str]:
    """Each check is a query returning the rows that break a rule, named by a '-- check:' line."""
    failures = []
    for statement in statements(path):
        match = re.search(r"--\s*check:\s*(.+)", statement)
        if not match:
            raise ValueError(f"{path.name}: every check needs a '-- check: <name>' line")
        name = match.group(1).strip()
        cursor.execute(statement)
        rows = cursor.fetchall()
        if rows:
            failures.append(f"{name}: {len(rows)} rows, e.g. {rows[:3]}")
    return failures
