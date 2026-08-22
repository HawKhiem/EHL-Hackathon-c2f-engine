from pathlib import Path

from c2f.extract import load_case, parse_items, parse_meta

CASE0 = Path(__file__).resolve().parents[1] / "cases" / "case_00"

SAMPLE = """Invoice
INVOICE NO.
2026-0001
DATE
6 Jan 2026
TRADE
Bikeshop
FROM
Bikey Bike Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 New Bike 1 unit
2 Labour for fitting the new
rear wheel 2.5 hours
3 Brake pads 4 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""


def test_parse_items_sample():
    items = parse_items(SAMPLE)
    assert [i["index"] for i in items] == [1, 2, 3]
    assert items[0] == {"index": 1, "description": "New Bike", "quantity": 1.0, "unit": "unit"}
    assert items[1]["quantity"] == 2.5 and items[1]["unit"] == "hours"
    assert "rear wheel" in items[1]["description"]
    assert items[2]["unit"] == "pcs"


def test_parse_items_unknown_layout_returns_empty():
    assert parse_items("nothing like an invoice here") == []


def test_parse_meta():
    m = parse_meta(SAMPLE)
    assert m["trade"] == "Bikeshop"
    assert m["vendor"] == "Bikey Bike Ltd"
    assert m["date"] == "6 Jan 2026"


def test_load_case_0():
    if not CASE0.exists():
        import pytest

        pytest.skip("case_00 not extracted")
    c = load_case(CASE0, 0)
    assert "BICYCLE THEFT" in c["policy"]
    assert "420" in c["description"]
    assert c["items"] == [{"index": 1, "description": "New Bike", "quantity": 1.0, "unit": "unit"}]
    assert c["invoice_meta"]["trade"] == "Bikeshop"
    assert c["images"] == []
