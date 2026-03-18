"""
NCAA March Mania 2026 — Phase 0 Hostile Dataset Audit.

Read-only adversarial audit of Kaggle raw CSV files to validate:
- LAW 3: Temporal Integrity
- LAW 4: Day 133 Tournament Boundary
- LAW 6: Data Completeness and Signal Integrity

Run from project root:
  ./.venv/bin/python scripts/dataset_audit.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "audit_report.md"

FATAL = "FATAL"
CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

REGULAR_SEASON_MAX_DAY = 132
TOURNEY_MIN_DAY = 134

findings: list[dict[str, str]] = []


def record(severity: str, task: str, message: str) -> None:
    findings.append({"severity": severity, "task": task, "message": message})
    level = {FATAL: logging.ERROR, CRITICAL: logging.ERROR, WARNING: logging.WARNING, INFO: logging.INFO}[severity]
    log.log(level, "[%s] %s — %s", severity, task, message)


def load_csv(filename: str) -> pd.DataFrame | None:
    path = RAW_DIR / filename
    if not path.exists():
        record(WARNING, "FILE_LOAD", f"Missing file: {filename}")
        return None
    try:
        log.info("Loading %s (%s bytes)", filename, path.stat().st_size)
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        record(WARNING, "FILE_LOAD", f"Empty CSV file: {filename}")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        record(CRITICAL, "FILE_LOAD", f"Failed to read {filename}: {exc}")
        return None


def ensure_columns(df: pd.DataFrame, required: list[str], filename: str, task: str) -> bool:
    missing = [col for col in required if col not in df.columns]
    if missing:
        record(CRITICAL, task, f"{filename}: missing required columns {missing}")
        return False
    return True


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def audit_temporal_horizon() -> tuple[list[str], list[list[str]]]:
    task = "TEMPORAL"
    lines: list[str] = ["## 2. TEMPORAL INTEGRITY TABLE", ""]

    regular_files = [
        "MRegularSeasonCompactResults.csv",
        "MRegularSeasonDetailedResults.csv",
        "WRegularSeasonCompactResults.csv",
        "WRegularSeasonDetailedResults.csv",
    ]
    tourney_files = ["MNCAATourneyCompactResults.csv", "WNCAATourneyCompactResults.csv"]

    temporal_rows: list[list[str]] = []
    boundary_violations: list[list[str]] = []

    for filename in regular_files:
        df = load_csv(filename)
        if df is None or not ensure_columns(df, ["Season", "DayNum"], filename, task):
            continue
        if df.empty:
            record(WARNING, task, f"{filename}: dataset is empty")
            continue

        season_range = (
            df.groupby("Season", as_index=False)["DayNum"]
            .agg(EarliestDay="min", LatestDay="max")
            .sort_values("Season")
        )

        violations = df[df["DayNum"] > REGULAR_SEASON_MAX_DAY]
        if not violations.empty:
            record(CRITICAL, task, f"{filename}: {len(violations)} rows with DayNum > {REGULAR_SEASON_MAX_DAY}")
            for idx, row in violations.sort_values(["Season", "DayNum"]).iterrows():
                boundary_violations.append([
                    filename,
                    str(int(row["Season"])),
                    str(int(row["DayNum"])),
                    str(int(idx)),
                    CRITICAL,
                    f"Regular season DayNum > {REGULAR_SEASON_MAX_DAY}",
                ])
        else:
            record(INFO, task, f"{filename}: no regular-season DayNum > {REGULAR_SEASON_MAX_DAY}")

        for _, row in season_range.iterrows():
            season = int(row["Season"])
            earliest = int(row["EarliestDay"])
            latest = int(row["LatestDay"])
            sev = "None"
            note = "None"
            if latest < REGULAR_SEASON_MAX_DAY:
                sev = WARNING
                note = f"Season ends before Day {REGULAR_SEASON_MAX_DAY}"
                record(WARNING, task, f"{filename}: season {season} ends on Day {latest}")
            temporal_rows.append([str(season), str(earliest), str(latest), f"{sev}: {note}"])

    for filename in tourney_files:
        df = load_csv(filename)
        if df is None or not ensure_columns(df, ["Season", "DayNum"], filename, task):
            continue
        if df.empty:
            record(WARNING, task, f"{filename}: dataset is empty")
            continue

        early_games = df[df["DayNum"] <= 133]
        if not early_games.empty:
            record(FATAL, task, f"{filename}: {len(early_games)} tournament rows with DayNum <= 133")
            for idx, row in early_games.sort_values(["Season", "DayNum"]).iterrows():
                boundary_violations.append([
                    filename,
                    str(int(row["Season"])),
                    str(int(row["DayNum"])),
                    str(int(idx)),
                    FATAL,
                    "Tournament DayNum <= 133",
                ])
        else:
            record(INFO, task, f"{filename}: all tournament rows satisfy DayNum >= {TOURNEY_MIN_DAY}")

    temporal_rows.sort(key=lambda r: (int(r[0]), r[3], int(r[1]), int(r[2])))
    lines.extend(markdown_table(["Season", "Earliest Day", "Latest Day", "Boundary Violations"], temporal_rows))
    lines.append("")

    lines.append("### Boundary Violation Detail")
    lines.append("")
    if boundary_violations:
        lines.extend(
            markdown_table(
                ["File", "Season", "DayNum", "GameID/Index", "Severity", "Issue"],
                sorted(boundary_violations, key=lambda r: (r[4], r[0], int(r[1]), int(r[2]), int(r[3]))),
            )
        )
    else:
        lines.append("No boundary violations detected in regular season or tournament files.")
    lines.append("")

    return lines, temporal_rows


def audit_massey_ordinals() -> list[str]:
    task = "MASSEY"
    lines: list[str] = ["## 3. MASSEY ORDINAL SPARSITY TABLE", ""]

    filename = "MMasseyOrdinals.csv"
    df = load_csv(filename)
    if df is None:
        lines.append("Massey file unavailable. Section skipped.")
        lines.append("")
        return lines
    if not ensure_columns(df, ["Season", "SystemName", "RankingDayNum"], filename, task):
        lines.append("Massey file missing required columns. Section partially skipped.")
        lines.append("")
        return lines
    if df.empty:
        record(WARNING, task, f"{filename}: dataset is empty")
        lines.append("Massey file is empty. Section skipped.")
        lines.append("")
        return lines

    day_freq = (
        df.groupby("RankingDayNum", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("RankingDayNum")
    )

    lines.append("### RankingDayNum Frequency Distribution")
    lines.append("")
    rows = [[str(int(r["RankingDayNum"])), str(int(r["Count"]))] for _, r in day_freq.iterrows()]
    lines.extend(markdown_table(["RankingDayNum", "Count"], rows))
    lines.append("")

    stats = (
        df.groupby("SystemName", as_index=False)["RankingDayNum"]
        .agg(EarliestDay="min", LatestDay="max")
        .sort_values("SystemName")
    )
    systems_day133 = set(df.loc[df["RankingDayNum"] == 133, "SystemName"].unique().tolist())

    system_rows: list[list[str]] = []
    for _, row in stats.iterrows():
        system = str(row["SystemName"])
        earliest = int(row["EarliestDay"])
        latest = int(row["LatestDay"])
        has_day133 = system in systems_day133

        if has_day133:
            status = "OK"
        elif latest < 128:
            status = f"{CRITICAL}: Final ranking before Day 128"
            record(CRITICAL, task, f"{system}: latest ranking Day {latest} (<128)")
        elif 128 <= latest <= 132:
            status = f"{WARNING}: Missing Day 133, latest in 128-132 (Day {latest})"
            record(WARNING, task, f"{system}: missing Day 133, latest ranking Day {latest}")
        else:
            status = f"{INFO}: Missing Day 133 but has later day ({latest})"
            record(INFO, task, f"{system}: missing Day 133 but has latest ranking Day {latest}")

        system_rows.append([
            system,
            str(earliest),
            str(latest),
            "Yes" if has_day133 else "No",
            status,
        ])

    lines.append("### System Coverage")
    lines.append("")
    lines.extend(markdown_table(["SystemName", "EarliestDay", "LatestDay", "HasDay133", "Status"], system_rows))
    lines.append("")

    missing_day133 = sorted(set(stats["SystemName"]) - systems_day133)
    lines.append("### Systems Missing Day 133: Boundary Fallback (128-132)")
    lines.append("")
    if missing_day133:
        fallback_rows: list[list[str]] = []
        for system in missing_day133:
            subset = df[df["SystemName"] == system]
            boundary_subset = subset[(subset["RankingDayNum"] >= 128) & (subset["RankingDayNum"] <= 132)]
            fallback = "NONE"
            if not boundary_subset.empty:
                fallback = str(int(boundary_subset["RankingDayNum"].max()))
                record(INFO, task, f"{system}: no Day 133, fallback day {fallback}")
            fallback_rows.append([system, fallback])
        lines.extend(markdown_table(["SystemName", "LatestAvailableDay(128-132)"], fallback_rows))
    else:
        lines.append("All systems provide Day 133 rankings.")
    lines.append("")

    lines.append(
        f"Systems providing Day 133 rankings: {len(systems_day133)} / {int(stats['SystemName'].nunique())}"
    )
    lines.append("")

    return lines


def audit_boxscore_quality() -> list[str]:
    task = "BOXSCORE"
    lines: list[str] = ["## 4. BOX SCORE SIGNAL QUALITY", ""]

    files = {
        "M": "MRegularSeasonDetailedResults.csv",
        "W": "WRegularSeasonDetailedResults.csv",
    }
    era_splits = {
        "M": [("2003-2010", 2003, 2010), ("2011-2025", 2011, 2025)],
        "W": [("2010-2015", 2010, 2015), ("2016-2025", 2016, 2025)],
    }
    modern_eras = {"M": "2011-2025", "W": "2016-2025"}
    box_cols = [
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
    ]

    flagged_rows: list[list[str]] = []

    for gender in ["M", "W"]:
        filename = files[gender]
        df = load_csv(filename)
        if df is None:
            continue
        if not ensure_columns(df, ["Season"], filename, task):
            continue
        if df.empty:
            record(WARNING, task, f"{filename}: dataset is empty")
            continue

        missing = [c for c in box_cols if c not in df.columns]
        if missing:
            record(WARNING, task, f"{filename}: missing expected box-score columns {missing}")
        available = sorted([c for c in box_cols if c in df.columns])

        lines.append(f"### {gender} Era Diagnostics ({filename})")
        lines.append("")
        for era_name, start, end in era_splits[gender]:
            era_df = df[(df["Season"] >= start) & (df["Season"] <= end)]
            if era_df.empty:
                lines.append(f"- Era {era_name}: no rows")
                continue

            lines.append(f"- Era {era_name}: {len(era_df)} rows")
            for col in available:
                nan_pct = float(era_df[col].isna().mean() * 100)
                zero_pct = float((era_df[col] == 0).mean() * 100)
                combined = nan_pct + zero_pct
                if era_name == modern_eras[gender] and combined > 10:
                    sev = CRITICAL if combined > 20 else WARNING
                    flagged_rows.append([
                        gender,
                        era_name,
                        col,
                        f"{nan_pct:.2f}",
                        f"{zero_pct:.2f}",
                        f"{combined:.2f}",
                        sev,
                    ])
                    record(sev, task, f"{filename} {era_name} {col}: NaN+Zero={combined:.2f}%")
        lines.append("")

    lines.append("### Columns Exceeding Modern-Era Threshold (NaN% + Zero% > 10)")
    lines.append("")
    if flagged_rows:
        flagged_rows = sorted(flagged_rows, key=lambda r: (r[6], r[0], r[1], r[2]))
        lines.extend(markdown_table(["Gender", "Era", "Column", "NaN%", "Zero%", "Combined%", "Severity"], flagged_rows))
    else:
        lines.append("No box-score columns exceeded threshold in modern eras.")
    lines.append("")

    return lines


def _extract_seed_refs(slots_df: pd.DataFrame) -> set[str]:
    refs: set[str] = set()
    for col in ["StrongSeed", "WeakSeed"]:
        if col not in slots_df.columns:
            continue
        candidate = slots_df[col].astype(str)
        mask = candidate.str.match(r"^[A-Z]\d{2}[ab]?$", na=False)
        refs.update(candidate[mask].tolist())
    return refs


def audit_seed_slot_integrity() -> list[str]:
    task = "SEED_SLOT"
    lines: list[str] = ["## 5. SEED / SLOT CONSISTENCY CHECK", ""]

    cfgs = [
        ("M", "MNCAATourneySeeds.csv", "MNCAATourneySlots.csv", "MNCAATourneyCompactResults.csv"),
        ("W", "WNCAATourneySeeds.csv", "WNCAATourneySlots.csv", "WNCAATourneyCompactResults.csv"),
    ]

    suffix_missing_rows: list[list[str]] = []
    drift_rows: list[list[str]] = []

    for gender, seeds_file, slots_file, tourney_file in cfgs:
        seeds_df = load_csv(seeds_file)
        slots_df = load_csv(slots_file)
        tourney_df = load_csv(tourney_file)

        lines.append(f"### {gender} Dataset")
        lines.append("")

        if seeds_df is None or not ensure_columns(seeds_df, ["Season", "Seed", "TeamID"], seeds_file, task):
            lines.append("Seeds file unavailable or invalid; checks skipped.")
            lines.append("")
            continue
        if seeds_df.empty:
            record(WARNING, task, f"{seeds_file}: dataset is empty")
            lines.append("Seeds file empty; checks skipped.")
            lines.append("")
            continue

        seeds_df = seeds_df.copy()
        seeds_df["SeedStr"] = seeds_df["Seed"].astype(str)

        suffix_rows = seeds_df[seeds_df["SeedStr"].str.match(r"^[A-Z]\d{2}[ab]$", na=False)]
        lines.append(f"Suffix seeds detected (`a`/`b`): {len(suffix_rows)}")

        if slots_df is not None and ensure_columns(slots_df, ["Season", "Slot", "StrongSeed", "WeakSeed"], slots_file, task):
            slot_seed_refs = _extract_seed_refs(slots_df)
            for season in sorted(slots_df["Season"].dropna().unique().tolist()):
                season = int(season)
                season_seed_values = set(seeds_df.loc[seeds_df["Season"] == season, "SeedStr"].unique().tolist())
                season_slots = slots_df[slots_df["Season"] == season]
                for _, slot_row in season_slots.iterrows():
                    for col in ["StrongSeed", "WeakSeed"]:
                        ref = str(slot_row[col])
                        if ref in slot_seed_refs and ref.endswith(("a", "b")) and ref not in season_seed_values:
                            suffix_missing_rows.append([
                                gender,
                                str(season),
                                str(slot_row["Slot"]),
                                col,
                                ref,
                                CRITICAL,
                            ])
                            record(CRITICAL, task, f"{gender} season {season}: slot {slot_row['Slot']} {col}={ref} missing from seeds")
        else:
            lines.append("Slots file unavailable or invalid; suffix join check skipped.")

        if tourney_df is not None and ensure_columns(tourney_df, ["Season", "WTeamID", "LTeamID"], tourney_file, task):
            for season in sorted(seeds_df["Season"].dropna().unique().tolist()):
                season = int(season)
                season_seeded = seeds_df[seeds_df["Season"] == season]
                season_games = tourney_df[tourney_df["Season"] == season]
                played = set(season_games["WTeamID"].dropna().astype(int)).union(
                    set(season_games["LTeamID"].dropna().astype(int))
                )
                for _, row in season_seeded.iterrows():
                    team_id = int(row["TeamID"])
                    if team_id not in played:
                        drift_rows.append([gender, str(season), str(row["Seed"]), str(team_id), WARNING])
                        record(WARNING, task, f"{gender} season {season}: seeded team {team_id} ({row['Seed']}) has no tourney result")
        else:
            lines.append("Tournament file unavailable or invalid; seed drift check skipped.")

        lines.append("")

    lines.append("### Missing `a`/`b` Seed References in Slots")
    lines.append("")
    if suffix_missing_rows:
        lines.extend(
            markdown_table(
                ["Gender", "Season", "Slot", "Column", "SeedRef", "Severity"],
                sorted(suffix_missing_rows, key=lambda r: (r[0], int(r[1]), r[2], r[3], r[4])),
            )
        )
    else:
        lines.append("No missing suffix seed references detected.")
    lines.append("")

    lines.append("### Seed Drift (Seeded Team Never Appears in Tournament Results)")
    lines.append("")
    if drift_rows:
        lines.extend(
            markdown_table(
                ["Gender", "Season", "Seed", "TeamID", "Severity"],
                sorted(drift_rows, key=lambda r: (r[0], int(r[1]), r[2], int(r[3]))),
            )
        )
    else:
        lines.append("No seed drift detected.")
    lines.append("")

    return lines


def generate_executive_summary() -> list[str]:
    lines = ["## 1. EXECUTIVE SUMMARY", ""]
    for severity in [FATAL, CRITICAL, WARNING, INFO]:
        scoped = [f for f in findings if f["severity"] == severity]
        lines.append(f"### {severity}")
        if scoped:
            for item in scoped:
                lines.append(f"- [{item['task']}] {item['message']}")
        else:
            lines.append("- None")
        lines.append("")
    return lines


def generate_final_recommendation() -> list[str]:
    lines = ["## 6. FINAL RECOMMENDATION", ""]
    fatal_temporal = [f for f in findings if f["task"] == "TEMPORAL" and f["severity"] == FATAL]
    critical_temporal = [f for f in findings if f["task"] == "TEMPORAL" and f["severity"] == CRITICAL]
    warning_temporal = [f for f in findings if f["task"] == "TEMPORAL" and f["severity"] == WARNING]

    if fatal_temporal:
        lines.append("Global Day 133 cutoff is NOT safe. Tournament boundary is violated (FATAL findings).")
    elif critical_temporal:
        lines.append("Global Day 133 cutoff is conditionally unsafe. Regular-season overflow past Day 132 exists (CRITICAL findings).")
    else:
        lines.append("Global Day 133 cutoff is structurally safe for tournament separation in this dataset.")

    if critical_temporal or fatal_temporal:
        lines.append("Recommendation: use season-adjusted safeguards before modeling.")
    elif warning_temporal:
        lines.append("Recommendation: keep global cutoff, but monitor seasons ending before Day 132 and enforce feature-window guards.")
    else:
        lines.append("Recommendation: global cutoff can remain fixed at Day 133 without season adjustment.")

    lines.append("")
    return lines


def generate_report(sections: list[list[str]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# NCAA March Mania 2026 — Hostile Raw Dataset Audit",
        "",
        "Generated by `scripts/dataset_audit.py`",
        "",
    ]

    lines.extend(generate_executive_summary())
    lines.append("---")
    lines.append("")

    for idx, sec in enumerate(sections):
        lines.extend(sec)
        if idx < len(sections) - 1:
            lines.append("---")
            lines.append("")

    lines.extend(generate_final_recommendation())

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written: %s", REPORT_PATH)


def main() -> None:
    log.info("Starting Phase 0 hostile dataset audit")
    if not RAW_DIR.exists():
        log.error("Missing raw directory: %s", RAW_DIR)
        sys.exit(1)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        log.error("No CSV files found in %s", RAW_DIR)
        sys.exit(1)

    log.info("Detected %d CSV files in %s", len(csv_files), RAW_DIR)

    temporal_lines, _ = audit_temporal_horizon()
    massey_lines = audit_massey_ordinals()
    boxscore_lines = audit_boxscore_quality()
    seed_slot_lines = audit_seed_slot_integrity()

    sections = [temporal_lines, massey_lines, boxscore_lines, seed_slot_lines]
    generate_report(sections)

    for severity in [FATAL, CRITICAL, WARNING, INFO]:
        count = sum(1 for f in findings if f["severity"] == severity)
        log.info("%s findings: %d", severity, count)

    if any(f["severity"] == FATAL for f in findings):
        sys.exit(2)


if __name__ == "__main__":
    main()
