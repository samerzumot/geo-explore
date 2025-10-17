from __future__ import annotations

def test_import_entity_extractor():
    from geoextract.extraction.entity_extractor import extract_entities_mvp
    assert callable(extract_entities_mvp)
