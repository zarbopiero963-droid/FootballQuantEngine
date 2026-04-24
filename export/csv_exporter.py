"""
CSV exporter for Football Quant Engine output.

Features
--------
- All pipeline fields exported (not just 4)
- Multiple export modes: "bets" (BET/WATCHLIST only), "full" (all rows), "summary"
- Date-stamped and auto-named output files
- Append mode for incremental daily accumulation
- Metadata header comment row (skippable by parsers)
- Unicode-safe (UTF-8 BOM for Excel compatibility)

Usage
-----
    from export.csv_exporter import CsvExporter

    exp = CsvExporter(output_dir="outputs")
    path = exp.export(records, mode="bets")          # filtered
    path = exp.export(records, mode="full")          # all rows
    path = exp.export_summary(records)               # league breakdown
    path = exp.append(new_records, "value_bets.csv") # append mode
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

# Canonical field order for value-bet exports
_BET_FIELDS = [
    "match_id",
    "market",
    "probability",
    "odds",
    "ev",
    "kelly",
    "kelly_stake",
    "tier",
    "model_edge",
    "market_edge",
    "confidence",
    "agreement",
    "decision",
    "over_25",
    "btts_yes",
]

_SUMMARY_FIELDS = [
    "league",
    "bets",
    "avg_ev",
    "avg_kelly",
    "bet_rate",
]

ExportMode = Literal["bets", "full", "summary"]


def _safe(v) -> str:
    """Convert any value to a clean CSV string."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _fieldnames_for(records: list[dict], preferred: list[str]) -> list[str]:
    """
    Return the preferred field order, supplemented by any extra keys
    actually present in the records (sorted, appended at end).
    """
    all_keys: set[str] = set()
    for r in records[:200]:  # sample first 200 rows
        all_keys.update(r.keys())

    ordered = [f for f in preferred if f in all_keys]
    extras = sorted(all_keys - set(preferred))
    return ordered + extras


class CsvExporter:
    """
    Multi-mode CSV exporter.

    Parameters
    ----------
    output_dir : base directory for output files (created if needed)
    utf8_bom   : write UTF-8 BOM so Excel auto-detects encoding (default True)
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        utf8_bom: bool = True,
    ) -> None:
        self._dir = Path(output_dir)
        self._utf8_bom = utf8_bom
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Primary export
    # ------------------------------------------------------------------

    def export(
        self,
        records: list[dict],
        mode: ExportMode = "bets",
        filename: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> str:
        """
        Export records to CSV.

        Parameters
        ----------
        records  : list of bet dicts (from PredictionPipeline or MatchRanker)
        mode     : "bets"    → only BET/WATCHLIST decisions
                   "full"    → all rows
                   "summary" → league-level aggregation
        filename : explicit filename (overrides auto-naming)
        date_str : date label for auto-naming (default: today)

        Returns
        -------
        str — absolute path of the written file.
        """
        today = date_str or date.today().isoformat()

        if mode == "summary":
            return self.export_summary(records, filename=filename, date_str=today)

        # Filter
        if mode == "bets":
            rows = [
                r
                for r in records
                if str(r.get("decision", "")).upper() in ("BET", "WATCHLIST")
            ]
        else:
            rows = list(records)

        if not filename:
            filename = (
                f"value_bets_{today}.csv"
                if mode == "bets"
                else f"full_export_{today}.csv"
            )

        path = self._dir / filename
        fields = _fieldnames_for(rows or records, _BET_FIELDS)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(
            path, "w", newline="", encoding="utf-8-sig" if self._utf8_bom else "utf-8"
        ) as f:
            # Metadata comment header
            f.write(f"# Football Quant Engine — {mode.upper()} export\n")
            f.write(f"# Generated: {ts}  |  Rows: {len(rows)}\n")

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\r\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _safe(row.get(k)) for k in fields})

        return str(path.resolve())

    # ------------------------------------------------------------------
    # Legacy compatibility (used by DashboardView and tests)
    # ------------------------------------------------------------------

    def export_value_bets(self, filepath: str, value_bets: list[dict]) -> None:
        """Write value bets to an explicit filepath (backwards-compatible)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = _fieldnames_for(value_bets, _BET_FIELDS)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(
            path, "w", newline="", encoding="utf-8-sig" if self._utf8_bom else "utf-8"
        ) as f:
            f.write("# Football Quant Engine — value bets export\n")
            f.write(f"# Generated: {ts}  |  Rows: {len(value_bets)}\n")
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\r\n",
            )
            writer.writeheader()
            for row in value_bets:
                writer.writerow({k: _safe(row.get(k)) for k in fields})

    # ------------------------------------------------------------------
    # Summary export
    # ------------------------------------------------------------------

    def export_summary(
        self,
        records: list[dict],
        filename: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> str:
        """
        Aggregate by league and export a one-row-per-league summary.
        """
        from collections import defaultdict

        today = date_str or date.today().isoformat()
        filename = filename or f"summary_{today}.csv"
        path = self._dir / filename

        acc: dict[str, dict] = defaultdict(
            lambda: {"bets": 0, "ev_sum": 0.0, "kelly_sum": 0.0, "bet_cnt": 0}
        )
        for r in records:
            league = str(
                r.get("league") or r.get("extras", {}).get("league") or "Unknown"
            )
            acc[league]["bets"] += 1
            acc[league]["ev_sum"] += float(r.get("ev", 0.0))
            acc[league]["kelly_sum"] += float(r.get("kelly", 0.0))
            acc[league]["bet_cnt"] += 1 if r.get("decision") == "BET" else 0

        rows = []
        for league, a in sorted(acc.items()):
            n = a["bets"] or 1
            rows.append(
                {
                    "league": league,
                    "bets": a["bets"],
                    "avg_ev": round(a["ev_sum"] / n, 6),
                    "avg_kelly": round(a["kelly_sum"] / n, 6),
                    "bet_rate": round(a["bet_cnt"] / n, 4),
                }
            )

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(
            path, "w", newline="", encoding="utf-8-sig" if self._utf8_bom else "utf-8"
        ) as f:
            f.write("# Football Quant Engine — league summary\n")
            f.write(f"# Generated: {ts}  |  Leagues: {len(rows)}\n")
            writer = csv.DictWriter(
                f,
                fieldnames=_SUMMARY_FIELDS,
                extrasaction="ignore",
                lineterminator="\r\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _safe(row.get(k)) for k in _SUMMARY_FIELDS})

        return str(path.resolve())

    # ------------------------------------------------------------------
    # Append mode
    # ------------------------------------------------------------------

    def append(self, new_records: list[dict], filename: str) -> str:
        """
        Append records to an existing CSV (or create it).

        Useful for building a cumulative daily bet log.
        """
        path = self._dir / filename
        exists = path.exists()
        fields = _fieldnames_for(new_records, _BET_FIELDS)

        with open(
            path,
            "a",
            newline="",
            encoding="utf-8-sig" if (self._utf8_bom and not exists) else "utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\r\n",
            )
            if not exists:
                writer.writeheader()
            for row in new_records:
                writer.writerow({k: _safe(row.get(k)) for k in fields})

        return str(path.resolve())

    # ------------------------------------------------------------------
    # In-memory helper (for Telegram attachments etc.)
    # ------------------------------------------------------------------

    def to_bytes(self, records: list[dict], mode: ExportMode = "bets") -> bytes:
        """Return CSV content as bytes without writing to disk."""
        if mode == "bets":
            rows = [r for r in records if r.get("decision") in ("BET", "WATCHLIST")]
        else:
            rows = list(records)

        fields = _fieldnames_for(rows or records, _BET_FIELDS)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe(row.get(k)) for k in fields})
        return buf.getvalue().encode("utf-8")
