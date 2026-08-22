# Copyright (c) QuantCo 2026-2026
# SPDX-License-Identifier: LicenseRef-QuantCo

# This starter script is a template for the most basic flow of the challenge.
# It lists the available games and uses the first one, the permanent test game
# that is always available for testing your API integration without waiting for
# scheduled games.
import os

import requests

BASE_URL = "https://c2f.public.quantco.cloud/"
TEAM_API_KEY = (
    os.environ.get("TEAM_API_KEY")
    or input(
        'Supply your team API key here, or set the environment variable "TEAM_API_KEY": '
    ).strip()
)

headers = {"X-API-Key": TEAM_API_KEY}


def section(title: str) -> None:
    """Print a section banner so the script output is easy to follow."""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def check(response: requests.Response, action: str) -> requests.Response:
    """Abort with a readable message if a request did not succeed."""
    if response.status_code != 200:
        print(f"  ✗ {action} failed: {response.status_code} - {response.text}")
        exit(1)
    return response


# ============================================================================
# PHASE 1: Retrieve the decryption key for the current game
# ============================================================================
section("PHASE 1 - Get the game and its decryption key")

# List all games
response = check(
    requests.get(f"{BASE_URL}/api/games/list", headers=headers), "Listing games"
)
games = response.json()
if not games:
    print("  ✗ No games available")
    exit(1)

print(f"  Found {len(games)} game(s):")
for game in games:
    print(f"    - game {game['id']} starts at {game['start_time']}")

# For now, we will use game 0, which is a permanent test game that is always available.
game_id = games[0]["id"]
print(f"  → Using game {game_id}")

# Get the decryption key
response = check(
    requests.get(f"{BASE_URL}/api/games/{game_id}/key", headers=headers),
    "Getting the decryption key",
)
decryption_key = response.json()["decryption_key"]
print(f"  ✓ Decryption key: {decryption_key}")

# Decrypt and extract the case archive using the decryption key. The archives are
# AES-256 encrypted zips, so we shell out to 7z (Python's zipfile can't read them).
# Case 0's archive is the test case shipped alongside this script.
import subprocess

case_zip = f"case_{game_id:02d}.zip"
case_dir = f"case_{game_id:02d}"
result = subprocess.run(
    ["7z", "x", "-y", f"-p{decryption_key}", f"-o{case_dir}", case_zip],
    capture_output=True,
    text=True,
    cwd="cases",
)
if result.returncode != 0:
    print(f"  ✗ Decrypting {case_zip} failed: {result.stdout}\n{result.stderr}")
    exit(1)
print(f"  ✓ Decrypted case files into ./cases/{case_dir}/")


# ============================================================================
# PHASE 2: Analyze the decrypted items, set the prices
# ============================================================================
section("PHASE 2 - Price the line items")

# TODO: Do your magic here: build a UI and do it manually? Randomly set prices?
#       Use a model to predict the best prices? It's up to you!
submissions = [
    {"index": 1, "charge_price": 410.0, "acceptance_limit": 430.0},
]

print(f"  ✓ Prepared {len(submissions)} submission(s)")


# ============================================================================
# PHASE 3: Submit your prices for the current game
# ============================================================================
section("PHASE 3 - Submit your prices")

response = check(
    requests.put(
        f"{BASE_URL}/api/games/{game_id}/submissions",
        headers=headers,
        json=submissions,
    ),
    "Submitting prices",
)
result = response.json()

print(f"  ✓ Successfully submitted {len(result)} line item(s):")
print(f"    {'line':>6}  {'charge price':>14}  {'acceptance limit':>18}")
for item in result:
    print(
        f"    {item['line_item_index']:>6}  {item['charge_price']:>14.2f}  "
        f"{item['acceptance_limit']:>18.2f}"
    )
print()
