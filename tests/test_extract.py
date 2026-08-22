from pathlib import Path

from c2f.extract import load_case, parse_meta, parse_pos, pdf_text

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
    # POS numbers only, advisory; the model still reads c["invoice_text"] itself.
    assert [it["index"] for it in c["items"]] == [1]
    assert c["items"][0]["description"] == "New Bike"
    assert "New Bike" in c["invoice_text"]
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



# Real pypdf output from game 27: the invoice that broke the old unit-list parser. "68 lines"
# was not a known unit, item 1 failed to match, and the all-or-nothing guard discarded all
# four items - so the case shipped with no chunk plan and no category labels.
GAME27 = """ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Translation from Spanish to English 68 lines
2 Material costs 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
INVOICE NO.
2026-0036
DUE
31 Jan 2026
TRADE
Compensation
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
3 Compensation for robbery damage 1 pcs
4 Shipping 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""

# Real pypdf output from game 28: item 1's quantity wrapped onto a line of its own as
# "12 hrs". Read as a POS number that is simply larger than the last one, it opens item 12
# and every item under it becomes part of its description.
GAME28 = """ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Electrician labour checking and rewiring the flooded
basement circuits
12 hrs
2 Vehicle costs 1 pcs
3 Procurement of a motor pump 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""

# Real pypdf output from game 11 (trimmed): POS numbers are NOT gap-free - the invoice runs
# 1..11 then 13..23 with no item 12, across two invoices in one PDF. A parser that demands
# exactly the next index drops twelve items; that cost two thirds of game 11's loss.
GAME11 = """ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Indoor leak detection 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
TRADE
Drying Technology
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
10 Room dryer unit 1 pcs
11 Drying fan 1 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
Invoice
TRADE
Plumbing
ITEMS
POS. DESCRIPTION AMOUNTUNIT TOTAL
13 Profipress elbow 90 copper 15mm model 01 5 pcs
14 Profipress elbow 90 copper 15mm model 02 5 pcs
INVOICE
Created on 21 Aug 2026 Page 1 / 1
"""


def idxs(text):
    return [it["index"] for it in parse_pos(text)]


def test_parse_pos_multi_invoice_and_wrapped_descriptions():
    items = parse_pos(GAME5)
    assert [it["index"] for it in items] == list(range(1, 18))
    by = {it["index"]: it["description"] for it in items}
    # a description wrapped over three lines is rejoined, and the trailing "1 pcs" is dropped
    assert by[1] == "Leak detection call-out and electro-acoustic pinpointing of the leak beneath the kitchen sink"
    assert by[3] == "Service technician hours"
    # "DUE / 5 Mar 2026" between invoices must not become item 5
    assert by[5].startswith("Freeing the affected pipe run")


def test_parse_pos_no_quantity_and_no_unit():
    # game 4: "- -" for quantity and unit, and a "flat rate" after a wrapped line
    assert idxs(GAME4) == list(range(1, 16))
    by = {it["index"]: it["description"] for it in parse_pos(GAME4)}
    assert by[5] == "HDMI cables and remote controls"
    assert by[6] == "Wall-mount bracket"


def test_parse_pos_unknown_unit_does_not_lose_the_invoice():
    # THE game 27 regression: "68 lines" is not a unit anyone listed, and it must not matter.
    items = parse_pos(GAME27)
    assert [it["index"] for it in items] == [1, 2, 3, 4]
    assert items[0]["description"] == "Translation from Spanish to English"


def test_parse_pos_wrapped_quantity_is_not_a_new_item():
    # THE game 28 regression: the orphaned "12 hrs" must stay part of item 1.
    items = parse_pos(GAME28)
    assert [it["index"] for it in items] == [1, 2, 3]
    assert items[0]["description"].startswith("Electrician labour checking and rewiring")
    assert items[1]["description"] == "Vehicle costs"


def test_parse_pos_keeps_gaps_in_the_numbering():
    # game 11: ... 10, 11, then 13. The gap is the invoice's business, not ours - we neither
    # renumber nor stop reading at it.
    assert idxs(GAME11) == [1, 10, 11, 13, 14]


def test_parse_pos_must_start_at_one():
    # A table that does not open at POS 1 means we misread the layout. No parse is the safe
    # answer: the model still gets the full invoice text, we only lose the chunk plan.
    assert parse_pos(GAME11.replace("1 Indoor leak detection 1 pcs", "7 Indoor leak detection 1 pcs")) == []


def test_parse_pos_returns_nothing_rather_than_guessing():
    assert parse_pos("no invoice table here at all") == []
    assert parse_pos("") == []
