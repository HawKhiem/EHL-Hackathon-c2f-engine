from pathlib import Path

from c2f.extract import load_case, parse_meta, pdf_text

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
    # The invoice is never line-parsed: the model reads c["invoice_text"] itself.
    assert c["items"] == []
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


