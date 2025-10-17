from __future__ import annotations

from pathlib import Path

def test_csv_writer(tmp_path: Path):
    from geoextract.export.csv_writer import write_csv

    data = {
        "coordinates": [{"format": "dd", "value": [40.0, -105.0], "crs": "EPSG:4326", "confidence": 0.9}],
        "assays": [{"element": "Au", "value": 1.2, "unit": "g/t", "confidence": 0.8}],
    }
    out = tmp_path / "out.csv"
    write_csv(data, out)
    assert out.exists()
    assert out.read_text().count("coordinate") == 1
