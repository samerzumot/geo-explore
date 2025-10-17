from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def write_csv(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for coord in data.get("coordinates", []):
        rows.append({
            "type": "coordinate",
            "format": coord.get("format"),
            "value": coord.get("value"),
            "crs": coord.get("crs"),
            "confidence": coord.get("confidence"),
            "context": coord.get("context"),
        })
    for assay in data.get("assays", []):
        rows.append({
            "type": "assay",
            "element": assay.get("element"),
            "value": assay.get("value"),
            "unit": assay.get("unit"),
            "detection_limit": assay.get("detection_limit"),
            "confidence": assay.get("confidence"),
            "context": assay.get("context"),
        })

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
