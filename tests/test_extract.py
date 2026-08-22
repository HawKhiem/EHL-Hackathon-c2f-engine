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


# Real pypdf output from game 5: four invoices in one PDF, quantities glued to the description
# ("sink1 pcs"), multi-word units ("flat rate"), wrapped descriptions, and a "DUE / 5 Mar 2026"
# header line between invoices that must not be mistaken for item 5.
GAME5 = """Invoice
INVOICE NO.
2026-0117
DATE
19 Feb 2026
DUE
5 Mar 2026
TRADE
Leak Detection
FROM
Drippy Dave Leak Hunters Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Leak detection call-out and electro-acoustic
pinpointing of the leak beneath the kitchen sink
1 pcs
2 Moisture measurement of the kitchen floor and wall
base
1 pcs
3 Service technician hours 14 hrs
4 Vehicle costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
INVOICE NO.
2026-0118
DATE
19 Feb 2026
DUE
5 Mar 2026
TRADE
Plumbing
FROM
Soggy Bottom Plumbing Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
5 Freeing the affected pipe run beneath the kitchen sink1 pcs
6 Removal and disposal of damaged pipe insulation 2 pcs
7 Repair of the confirmed leak on the copper supply pipe1 pcs
8 Replacement copper pipe section and transition fittings1 flat rate
9 Vehicle costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
INVOICE NO.
2026-0119
DUE
5 Mar 2026
TRADE
Drying Technology
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
10 Condensation dryer for the kitchen, rental for the
drying period
1 pcs
11 Room drying of the kitchen floor and wall base 1 pcs
12 Vehicle costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
TRADE
Carpentry
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
13 Removal, transport and disposal of the water-damaged
wooden kitchen table
3 pcs
14 Supply of replacement table - premium solid-oak
designer model, higher specification than the original
1 pcs
15 Delivery and assembly of the replacement table 1 flat rate
16 Cleaning of the installation area 1 pcs
17 Vehicle costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""

# Real pypdf output from game 4: "– –" (no quantity, no unit), "flat rate" after a wrapped line.
GAME4 = """ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 TV set (surge damaged) 1 pcs
2 Speaker system (surge damaged) 1 pcs
3 AV receiver / amplifier (surge damaged) 1 pcs
4 Melted mains plug and power lead (physically surge-
damaged)
1 flat rate
5 HDMI cables and remote controls – –
6 Wall-mount bracket – –
7 DVD player (was already failing before the storm, age-
related)
1 pcs
8 Router (no diagnostic report provided) – –
9 Shipping 1 pcs
10 Installation 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
DUE
5 Mar 2026
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
11 Diagnostic inspection and surge-failure report for
damaged electronics
2 pcs
12 Vehicle costs – return visit – –
13 Wiring safety check of property distribution board – –
14 Administrative and claim-processing fee 1 flat rate
15 Vehicle costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""


def test_parse_items_game5_multi_invoice_glued_qty_and_flat_rate():
    items = parse_items(GAME5)
    assert [i["index"] for i in items] == list(range(1, 18))
    by = {i["index"]: i for i in items}
    assert by[1]["description"].endswith("beneath the kitchen sink") and by[1]["quantity"] == 1 and by[1]["unit"] == "pcs"
    assert by[3] == {"index": 3, "description": "Service technician hours", "quantity": 14.0, "unit": "hrs"}
    assert by[5]["description"] == "Freeing the affected pipe run beneath the kitchen sink"
    assert by[5]["quantity"] == 1.0 and by[5]["unit"] == "pcs"
    assert by[8]["description"] == "Replacement copper pipe section and transition fittings"
    assert by[8]["quantity"] == 1.0 and by[8]["unit"] == "flat rate"
    assert by[13]["quantity"] == 3.0
    assert by[15]["unit"] == "flat rate"
    assert "Mar" not in by[5]["description"]


def test_parse_items_game4_dash_dash_and_flat_rate():
    items = parse_items(GAME4)
    assert [i["index"] for i in items] == list(range(1, 16))
    by = {i["index"]: i for i in items}
    assert by[4]["description"].endswith("(physically surge- damaged)") and by[4]["unit"] == "flat rate"
    assert by[5] == {"index": 5, "description": "HDMI cables and remote controls", "quantity": 0.0, "unit": "–"}
    assert by[7]["quantity"] == 1.0 and by[7]["unit"] == "pcs"
    assert by[12]["description"] == "Vehicle costs – return visit"
    assert by[14]["unit"] == "flat rate"


def test_parse_items_keeps_every_line_across_any_gap_in_the_numbering():
    """Game 11: 1..11 then 13..23, no item 12. Gaps are the invoice's business - and the jump
    is not capped: whatever the numbering does, every item line ends up in the list."""
    rows = [f"{i} Item {i} 1 pcs" for i in list(range(1, 12)) + list(range(13, 24)) + [80, 200]]
    text = "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n" + "\n".join(rows) + "\nINVOICE\n"
    items = parse_items(text)
    assert [i["index"] for i in items] == list(range(1, 12)) + list(range(13, 24)) + [80, 200]


def test_parse_items_wrapped_quantity_line_is_not_a_new_item():
    """No gap limit means "10 pcs" under item 2 could look like item 10; the glued-line test
    must keep it as item 2's quantity."""
    text = "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Sink 1 pcs\n2 Very long description of the tap\n10 pcs\n3 Hose 3 m\nINVOICE\n"
    items = parse_items(text)
    assert [i["index"] for i in items] == [1, 2, 3]
    assert items[1]["quantity"] == 10.0 and items[1]["unit"] == "pcs"


def test_parse_items_drops_a_truncated_parse_rather_than_pricing_half_an_invoice():
    """We never skip items: an item line the walker missed invalidates the whole parse, so
    run.py falls back to the model reading the raw invoice text."""
    text = (
        "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Sink 1 pcs\n5 Tap 1 pcs\n"
        "3 Bathtub 1 pcs\n"  # numbering goes backwards: unreachable, and unmistakably an item
        "INVOICE\n"
    )
    assert parse_items(text) == []
