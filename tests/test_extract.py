import json
from pathlib import Path

from c2f.extract import case_labels, item_labels, load_case, parse_meta, pdf_text

CASE0 = Path(__file__).resolve().parents[1] / "cases" / "case_00"

SAMPLE = """Invoice
INVOICE NO.
1000-0001
DATE
6 Jan 2000
TRADE
Tradeshop
FROM
Vendor C Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 New Item 1 unit
2 Labour for fitting the new
rear wheel 2.5 hours
3 Brake pads 4 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
"""


def test_parse_meta():
    m = parse_meta(SAMPLE)
    assert m["trade"] == "Tradeshop"
    assert m["vendor"] == "Vendor C Ltd"
    assert m["date"] == "6 Jan 2000"


def test_load_case_0():
    if not CASE0.exists():
        import pytest

        pytest.skip("case_00 not extracted")
    c = load_case(CASE0, 0)
    assert "BICYCLE THEFT" in c["policy"]
    assert "420" in c["description"]
    # The model is still never handed a parsed table - it reads c["invoice_text"] itself.
    # c["items"] mirrors the recovered POS numbers, and exists only so c2f.run can chunk.
    assert [it["index"] for it in c["items"]] == [1]
    assert c["items"][0]["description"] == "New Bike"
    assert "New Bike" in c["invoice_text"]
    assert c["invoice_meta"]["trade"] == "Bikeshop"
    assert c["images"] == []


# Real pypdf output from game 5: four invoices in one PDF, quantities glued to the description
# ("sink1 pcs"), multi-word units ("flat rate"), wrapped descriptions, and a "DUE / 5 Mar 2000"
# header line between invoices that must not be mistaken for item 5.
GAME5 = """Invoice
INVOICE NO.
1000-0117
DATE
19 Feb 2000
DUE
5 Mar 2000
TRADE
Leak Detection
FROM
Vendor A Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Service call-out and acoustic
locating of the fault beneath the fixture
1 pcs
2 Moisture measurement of the floor and wall
base
1 pcs
3 Service technician hours 14 hrs
4 Vehicle costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
Invoice
INVOICE NO.
1000-0118
DATE
19 Feb 2000
DUE
5 Mar 2000
TRADE
Plumbing
FROM
Vendor B Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
5 Freeing the affected pipe run beneath the fixture1 pcs
6 Removal and disposal of damaged pipe insulation 2 pcs
7 Repair of the confirmed leak on the copper supply pipe1 pcs
8 Replacement copper pipe section and transition fittings1 flat rate
9 Vehicle costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
Invoice
INVOICE NO.
1000-0119
DUE
5 Mar 2000
TRADE
Drying Technology
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
10 Condensation dryer for the room, rental for the
drying period
1 pcs
11 Room drying of the floor and wall base 1 pcs
12 Vehicle costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
Invoice
TRADE
Carpentry
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
13 Removal, transport and disposal of the damaged
wooden table
3 pcs
14 Supply of replacement table - premium
model, higher specification than the original
1 pcs
15 Delivery and assembly of the replacement table 1 flat rate
16 Cleaning of the installation area 1 pcs
17 Vehicle costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
"""

# Real pypdf output from game 4: "– –" (no quantity, no unit), "flat rate" after a wrapped line.
GAME4 = """ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Device A (damaged) 1 pcs
2 Device B (damaged) 1 pcs
3 Device C (damaged) 1 pcs
4 Damaged plug and power lead (physically
damaged)
1 flat rate
5 HDMI cables and remote controls – –
6 Wall-mount bracket – –
7 Device D (was already failing before, age-
related)
1 pcs
8 Device E (no diagnostic report provided) – –
9 Shipping 1 pcs
10 Installation 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
Invoice
DUE
5 Mar 2000
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
11 Diagnostic inspection and failure report for
damaged devices
2 pcs
12 Vehicle costs – return visit – –
13 Wiring safety check of distribution board – –
14 Administrative and claim-processing fee 1 flat rate
15 Vehicle costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
"""




# Real pypdf output from game 27: "68 lines" is not a unit the old table parser knew, and
# because that parser was all-or-nothing it threw away all four items -- which is what left
# games 27+ with no line labels at all (empty _description => no per-bucket bias, no market
# history). Label recovery must never be all-or-nothing.
GAME27 = """Invoice
INVOICE NO.
1000-0035
DATE
17 Jan 2000
DUE
31 Jan 2000
TRADE
Translation Services
FROM
Vendor D Ltd
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Translation service 68 lines
2 Material costs 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
Invoice
TRADE
Compensation
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
3 Compensation for reported damage 1 pcs
4 Shipping 1 pcs
INVOICE
Created on 1 Jan 2000 Page 1 / 1
"""


def test_item_labels_unknown_unit_keeps_every_item():
    """The game 27 regression: one unfamiliar unit must not discard the invoice."""
    labels = item_labels(GAME27)
    assert labels == {
        1: "Translation service",
        2: "Material costs",
        3: "Compensation for reported damage",
        4: "Shipping",
    }


def test_item_labels_strips_quantity_and_unit():
    labels = item_labels(SAMPLE)
    assert labels[1] == "New Item"
    assert labels[3] == "Brake pads"


def test_item_labels_joins_wrapped_descriptions():
    assert item_labels(SAMPLE)[2] == "Labour for fitting the new rear wheel"


def test_item_labels_handles_glued_quantity_and_multiword_unit():
    """pypdf glues the quantity onto the description ("sink1 pcs"); "flat rate" is two words."""
    labels = item_labels(GAME5)
    assert labels[5] == "Freeing the affected pipe run beneath the fixture"
    assert labels[8] == "Replacement copper pipe section and transition fittings"


def test_item_labels_reads_every_invoice_in_the_pdf():
    labels = item_labels(GAME5)
    assert sorted(labels) == list(range(1, 18))
    assert labels[3] == "Service technician hours"
    assert labels[17] == "Vehicle costs"


def test_item_labels_ignores_header_fields_between_invoices():
    """"DUE / 5 Mar 2000" sits between invoices and must not be read as item 5."""
    assert "Mar 2026" not in item_labels(GAME5)[5]


def test_item_labels_handles_dash_quantities():
    labels = item_labels(GAME4)
    assert labels[5] == "HDMI cables and remote controls"
    assert labels[12] == "Vehicle costs – return visit"


def test_item_labels_without_items_table():
    assert item_labels("just some prose, no invoice table here") == {}


def test_load_case_0_recovers_labels():
    if not CASE0.exists():
        import pytest

        pytest.skip("case_00 not extracted")
    c = load_case(CASE0, 0)
    # items and item_labels are the same recovery, in the two shapes their readers want:
    # c2f.run iterates items to plan chunks, c2f.price/history key on the labels.
    assert "New Bike" in c["item_labels"].values()
    assert {it["index"]: it["description"] for it in c["items"]} == {
        int(k): v for k, v in c["item_labels"].items()
    }


def test_item_labels_strips_quantity_with_thousands_separator():
    """Real game 17 line: "2,412.1kWh" must come off whole, not leave "costs 2," behind."""
    text = (
        "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n"
        "19 Material surcharge 1 flat rate\n"
        "20 Electricity costs 2,412.1kWh\n"
        "INVOICE\n"
    )
    assert item_labels(text)[20] == "Electricity costs"


def test_case_labels_reads_item_labels_through_a_json_round_trip():
    """Run logs are json, so the int keys come back as strings."""
    case = {"item_labels": {"1": "Vehicle costs", "2": "Helper hours"}}
    assert case_labels(json.loads(json.dumps(case))) == {1: "Vehicle costs", 2: "Helper hours"}


def test_case_labels_recovers_from_invoice_text_when_the_log_has_none():
    """Games 27-28 logged neither item_labels nor an items parse - only the invoice text."""
    case = {"items": [], "invoice_text": GAME27}
    assert case_labels(case)[1] == "Translation service"


def test_case_labels_falls_back_to_a_stored_items_parse():
    case = {"items": [{"index": 3, "description": "Skilled worker hours"}], "invoice_text": ""}
    assert case_labels(case) == {3: "Skilled worker hours"}


def test_case_labels_of_a_case_with_nothing_to_read():
    assert case_labels({}) == {}


def test_load_case_items_drive_chunking():
    """c2f.run plans its chunks off case["items"]; an empty list silently disables chunking.

    Games 29-33 hid this: 4-7 item invoices answer in one call in under 10 s either way. It
    surfaced on the big ones - game 31 (18 items) took 32.8 s and its submission came back
    403 GAME ALREADY ENDED, game 35 (20 items) took 37.9 s and only just landed.
    """
    from c2f.run import plan_chunks

    text = "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n" + "".join(
        f"{i} Item number {i} 1 pcs\n" for i in range(1, 21)
    )
    labels = item_labels(text)
    assert len(labels) == 20
    items = [{"index": i, "description": d} for i, d in sorted(labels.items())]
    chunks = plan_chunks([it["index"] for it in items], None)
    assert len(chunks) == 2, "20 items must split; one call is what lost game 31"
    assert sorted(i for c in chunks for i in c) == list(range(1, 21)), "no item may be dropped"
