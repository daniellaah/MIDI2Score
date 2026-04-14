from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "length_bucketing.md"
TSV_PATH = REPO_ROOT / "exp" / "apr12-lmx-batching.tsv"
START_MARKER = "<!-- sampler-research:start -->"
END_MARKER = "<!-- sampler-research:end -->"


@dataclass
class Record:
    commit: str
    valid_loss: float
    tokens_per_second: float
    status: str
    description: str


def main() -> None:
    records = load_records(TSV_PATH)
    replacement = build_research_section(records)
    document = DOC_PATH.read_text(encoding="utf-8")
    start = document.index(START_MARKER) + len(START_MARKER)
    end = document.index(END_MARKER)
    updated = document[:start] + "\n" + replacement + "\n" + document[end:]
    DOC_PATH.write_text(updated, encoding="utf-8")


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        commit, valid_loss, toks_per_sec, status, description = line.split("\t", maxsplit=4)
        records.append(
            Record(
                commit=commit,
                valid_loss=float(valid_loss),
                tokens_per_second=float(toks_per_sec),
                status=status,
                description=description,
            )
        )
    return records


def build_research_section(records: list[Record]) -> str:
    sampler_speed = [record for record in records if "[600s sampler-speed]" in record.description]
    canonical_7200 = [
        record
        for record in records
        if "[7200s" in record.description and "canonical]" in record.description
    ]
    sampler_grid = [record for record in records if "[1800s sampler-grid]" in record.description]
    fastest_sampler = max(
        (record for record in sampler_speed if record.status != "crash"),
        key=lambda record: record.tokens_per_second,
        default=None,
    )
    best_7200 = min(
        (record for record in canonical_7200 if record.status != "crash"),
        key=lambda record: record.valid_loss,
        default=None,
    )

    lines: list[str] = []
    lines.append("This section is synchronized from `exp/apr12-lmx-batching.tsv`.")
    lines.append("")
    lines.append("Selection rule:")
    lines.append("- `600s sampler-speed`: mark `keep` only for the highest `tokens/sec`; all other successful runs are `discard`.")
    lines.append("- `7200s canonical`: mark `keep` only for the lowest `validation loss`; all other successful runs are `discard`.")

    if sampler_speed:
        if fastest_sampler is not None:
            lines.append("")
            lines.append("Current fastest 600s sampler configuration:")
            lines.append(
                f"- `{clean_description(fastest_sampler.description)}` at `{fastest_sampler.tokens_per_second:.1f}` tokens/sec"
            )
            lines.append(f"- `best val loss = {fastest_sampler.valid_loss:.6f}`")

        lines.append("")
        lines.append("### 600s Sampler-Speed Runs")
        lines.extend(
            render_table(
                sampler_speed,
                status_override=lambda record: select_sampler_speed_status(record, fastest_sampler),
            )
        )

    if canonical_7200:
        lines.append("")
        lines.append("### 7200s Canonical Runs")
        lines.extend(
            render_table(
                canonical_7200,
                status_override=lambda record: select_canonical_7200_status(record, best_7200),
            )
        )
        if best_7200 is not None:
            lines.append("")
            lines.append("Current best 7200s canonical result:")
            lines.append(
                f"- `{clean_description(best_7200.description)}` with `val loss = {best_7200.valid_loss:.6f}`"
            )

    if sampler_grid:
        lines.append("")
        lines.append("### 1800s Sampler Grid Runs")
        lines.extend(
            render_table(
                sampler_grid[-12:],
                headers=["commit", "valid_loss", "toks/sec", "status", "description"],
            )
        )
        best_grid = min(
            (record for record in sampler_grid if record.status != "crash"),
            key=lambda record: record.valid_loss,
            default=None,
        )
        fastest_grid = max(
            (record for record in sampler_grid if record.status != "crash"),
            key=lambda record: record.tokens_per_second,
            default=None,
        )
        lines.append("")
        lines.append("Sampler-grid takeaways so far:")
        if best_grid is not None:
            lines.append(
                f"- Lowest validation loss so far: `{clean_description(best_grid.description)}` "
                f"with `{best_grid.valid_loss:.6f}`"
            )
        if fastest_grid is not None:
            lines.append(
                f"- Highest throughput so far: `{clean_description(fastest_grid.description)}` "
                f"at `{fastest_grid.tokens_per_second:.1f}` tokens/sec"
            )

    if not sampler_speed and not canonical_7200 and not sampler_grid:
        lines.append("")
        lines.append("No sampler research results have been logged yet.")

    return "\n".join(lines)


def render_table(records: list[Record], *, status_override) -> list[str]:
    lines = [
        "| commit | valid_loss | toks/sec | status | description |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.commit,
                    f"{record.valid_loss:.6f}",
                    f"{record.tokens_per_second:.1f}",
                    status_override(record),
                    f"`{clean_description(record.description)}`",
                ]
            )
            + " |"
        )
    return lines


def select_sampler_speed_status(record: Record, fastest: Record | None) -> str:
    if record.status == "crash":
        return "crash"
    if fastest is not None and record.description == fastest.description:
        return "keep"
    return "discard"


def select_canonical_7200_status(record: Record, best: Record | None) -> str:
    if record.status == "crash":
        return "crash"
    if best is not None and record.description == best.description:
        return "keep"
    return "discard"


def clean_description(description: str) -> str:
    return description.split(" run=")[0]


if __name__ == "__main__":
    main()
