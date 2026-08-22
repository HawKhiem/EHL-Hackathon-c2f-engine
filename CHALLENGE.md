# 🏆 Claim to Fame

Hello hackers, welcome to QuantCo's _Claim to Fame_ challenge!

---

## 🎭 Scenario

In insurances, a _claim_ is a request by a policyholder asking for payment or financial compensation. After property damage or a car accident, tradespeople such as handymen and repair garages issue invoices. Policyholders submit these invoices to their insurance providers, which decide whether to pay.

But not everyone is honest, so for each invoiced item the insurer must judge whether it is:

- **Covered** by the policy,
- **Related** to the reported damage, and
- **Reasonably** priced (we refer to "total" price per line item in this challenge = quantity * unit price)

A fraudulent claim fails one of these: it isn't insured, the item is unrelated to the case, or the price is inflated.

## 🎯 Your Task

In this challenge you're competing against other teams in rounds. Your goal is to **a) maximize the amount of money you make** and **b) do so with style**.

In this challenge, you will take on two roles simultaneously:

| Role                             | What you do                                                                                                                                                    |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧾 **Invoice Issuer (Handyman)** | Set the price (= quantity * unit price) for each line item on the invoice that you think you're entitled to. If you set a fair price, you will get your money! |
| 🔍 **Insurance Claim Handling**  | Review incoming invoices and minimize the amount your team pays. Spot fraudulent items and set a maximum fair price above which you reject payments.           |

In every round, you submit the prices a and b for all items, your submissions will be matched against all other teams.

> ⚠️ **But bear in mind:** you can't charge or pay whatever you like. There is a **fair value threshold `t`** for each item. If you reject a request that was at or below this value, you will get a penalty.

## 🎲 The Game

In regular intervals, a case will be released. You then have _exactly one minute_ to submit. Once the case is released, you will be able to fetch a _decryption key_ from the API. This can then be used to decrypt the `.zip` file for the case. Each case consists of:

| File              | Contents                                            |
| ----------------- | --------------------------------------------------- |
| `policy.txt`      | An **insurance policy** describing what is covered  |
| `description.txt` | A **damage description** explaining the claim       |
| `invoices.pdf`    | An **invoice** containing line items without prices |
| `images.png`      | Some cases might also contain **images**            |

After decrypting and extracting, you should analyze the files and set two values for every line item:

- **Charge price (`a`)**: the amount you charge opposing teams. _Default: `0`_
- **Acceptance limit (`b`)**: the maximum amount your team is willing to accept and pay when receiving the same line item from another team. _Default: `0`_

> 💶 **Always submit the _gross total_.** Both the charge price `a` and the acceptance limit `b` must always be given as the gross total for the whole line item. Never the net amount, and never a per-unit price.

As mentioned, there will be a **secret threshold (`t`) representing the maximum price that a claims expert would consider appropriate**. For line items that are **not covered** by the policy: `t = 0`.

### 💸 Payoffs

In the following: `H` is the handyman who issues the invoice, `I` is the insurance that does the claim handling and invoice checking.

Everything hinges on where the charge price `a` falls relative to the secret threshold `t`. Picture it on a number line:

```
        fair zone  (a ≤ t)         │      fraud zone  (a > t)
  ◀────────────────────────────────┼───────────────────────────────▶
  0                                t                                 a →
  the price is valid; H is owed it │ the price is inflated; H is not owed it
```

- **Left of `t`, the fair zone (`a ≤ t`).** The charge is legitimate, so `H` is entitled to the money. `I` _should_ accept: accepting pays the fair `a`, while rejecting still owes `H` the `a` **plus** a `0.5a` lawyer penalty on `I`.
- **Right of `t`, the fraud zone (`a > t`).** The charge is inflated, so `H` is not owed anything. `I` _should_ reject: rejecting means no money flows, while accepting hands `H` the fraudulent amount.

The threshold `t` is the dividing line between the two, and it's secret; both roles are really betting on which side of `t` each price sits.

> 🧢 **Accepted-payment cap.** Each line item has a secret payment cap c, shared across all teams, where `c ≥ 4t` (and never below an absolute floor). If a charge is accepted, the insurer pays and the issuer receives min(a,c). This prevents teams from winning through a single exorbitantly high charge.

The table below details what happens for each transaction.

|                            | `a ≤ t`: price okay               | `a > t`: price fraudulent                  |
| -------------------------- | --------------------------------- | ------------------------------------------ |
| **`a ≤ b` price accepted** | `I` pays `a`, `H` gets `a`        | `I` pays `min(a, c)`, `H` gets `min(a, c)` |
| **`a > b` price rejected** | `I` pays `a + 0.5a`, `H` gets `a` | `I` pays `0`, `H` gets `0`                 |

**Left column (`a ≤ t`): the price is valid.**

- If insurance `I` accepts (`a ≤ b`), it pays the claim: `I` pays `a`, `H` gets `a`.
- If insurance `I` wrongfully rejects (`a > b`), `H` still gets the `a` it is owed, but `I` also pays a lawyer as a penalty `0.5a`, so `I` pays `1.5a` in total.

**Right column (`a > t`): the price is fraudulent.**

- If insurance `I` accepts (`a ≤ b`), `H` is in luck and pockets the capped fraudulent amount `min(a, c)`.
- If insurance `I` rightfully rejects (`a > b`), no money flows.

## 🧮 An example round

Let's walk through a single round with three teams (**Alpha Squad**, **Beta Dynamics**, and **Delta Strategies**) and just **one line item**: a replacement windshield. Suppose the secret fair value threshold for this item is `t = 100`.

The game starts at 19:00:00. Teams then have exactly one minute to fetch the decryption key, decrypt the zip file, analyze it and submit they charge price `a` and acceptance limit `b`.
Suppose the teams submit:

| Team             | Charge `a` | Acceptance limit `b` |
| ---------------- | ---------- | -------------------- |
| Alpha Squad      | `100`      | `130`                |
| Beta Dynamics    | `150`      | `90`                 |
| Delta Strategies | `100`      | `110`                |

Every team is matched against every other team, in both roles. For each matchup, the **issuer** acts as the handyman `H` (charging its `a`) and the **reviewer** acts as the insurance `I` (deciding with its `b`). That gives six transactions:

| Issuer (H) → Reviewer (I) | `a`   | Reviewer's `b` | `a ≤ t`? | `a ≤ b`?  | Outcome                                                   |
| ------------------------- | ----- | -------------- | -------- | --------- | --------------------------------------------------------- |
| Alpha → Beta              | `100` | `90`           | ✅ okay  | ❌ reject | Wrongful reject: Beta pays `1.5a = 150`, Alpha gets `100` |
| Alpha → Delta             | `100` | `110`          | ✅ okay  | ✅ accept | Delta pays `100`, Alpha gets `100`                        |
| Delta → Alpha             | `100` | `130`          | ✅ okay  | ✅ accept | Alpha pays `100`, Delta gets `100`                        |
| Delta → Beta              | `100` | `90`           | ✅ okay  | ❌ reject | Wrongful reject: Beta pays `1.5a = 150`, Delta gets `100` |
| Beta → Alpha              | `150` | `130`          | ❌ fraud | ❌ reject | Rightful reject: no money flows                           |
| Beta → Delta              | `150` | `110`          | ❌ fraud | ❌ reject | Rightful reject: no money flows                           |

Adding up what each team **gets** (as customer) minus what it **pays** (as insurance):

| Team             | Gets  | Pays  | **Net**    |
| ---------------- | ----- | ----- | ---------- |
| Alpha Squad      | `200` | `100` | **`+100`** |
| Delta Strategies | `200` | `100` | **`+100`** |
| Beta Dynamics    | `0`   | `300` | **`−300`** |

The lesson: **Beta Dynamics loses on both sides.** As a customer it overcharged (`a = 150 > t`), so its fraudulent claims were rightfully rejected and it earned nothing. As an insurance it set its acceptance limit too low (`b = 90 < t`), so it wrongfully rejected fair claims and paid the `1.5a` lawyer penalty + compensation twice. Alpha and Delta both charged fairly and set acceptance limits above the threshold, so they collected their money and avoided penalties.

## ⚙️ Game Setup

To participate, you need to take multiple steps.

### 1. Register

First, register for the challenge. To do so, come up to us, tell us your team and Discord username to obtain your API key. Details will follow on Discord.

You will then obtain your key. This key is unique and unchangeable. **Do not share it** with anyone outside your team (duh), as they would be able to trade on your behalf.

### 2. Get the Cases

Later, we will provide a link to a folder. The folder will contain a number of encrypted `.zip` files, one for each case. There is a test case (case 0) in the folder. Together with the provided `starter_script.py`, you can use the endpoint to see if your requests are properly formatted. The endpoint accepts any well-formatted JSON. Further details can be found in [API_HANDBOOK.md](API_HANDBOOK.md) in the folder.

## 🚫 Fair play

This is a competition between teams, so anything that undermines that is off limits. This includes:

- **No cross-team collaboration.** Do not coordinate prices, acceptance limits, or strategies with other teams, and do not share your analysis of a case with them.
- **No sharing keys or credentials.** Your API key belongs to your team only. Never submit on behalf of another team, and never let another team submit on yours.
- **No attacking the platform.** Do not try to obtain decryption keys before a case is released, read other teams' submissions, extract the secret thresholds, or otherwise probe, overload, or exploit the API and its infrastructure.
- **Anything inside your own team is fair game.** Build tooling, use LLMs, do stuff manually, research the domain, ...

> ⚠️ Cheating gets your team disqualified from the tournament. If you are unsure whether something is allowed, just ask the organizers.

## 🥇 How to win

Shortly before the challenge ends, you are asked to provide a small write-up of your strategy. We will assess your approach and look at how well it performed to determine the winner. The best teams will present their approach to us.

## 🧰 What you need

Inside the shared folder, we include a starter script in Python to get you going, but you can use any language, library, or framework you'd like. The `starter_script.py` that already implements this flow for the mentioned test `Case 0`. Run it to get an idea of how things work.

To run it, you'll need to provide your API keys (`TEAM_API_KEY` and `OPENAI_KEY`) as environment variables — the script prompts you for the team API key if `TEAM_API_KEY` is not set — and install its dependencies: the Python packages `requests` and `openai`, plus **7-Zip**, which the script calls to decrypt the case archives.

**The easy way ([Pixi](https://pixi.sh)):** we ship a `pixi.toml` that pulls in Python, both packages, and 7-Zip in one go. From this folder just run:

```bash
pixi install
pixi run python starter_script.py
```

**The manual way:** in a plain Python environment, install the packages with `pip install requests openai`, then install 7-Zip so that the `7z` command is on your `PATH`:

- **macOS** (via [Homebrew](https://brew.sh)): `brew install p7zip`
- **Linux (Debian/Ubuntu):** `sudo apt install p7zip-full`
- **Windows:** download and run the installer from [7-zip.org/download.html](https://www.7-zip.org/download.html), then add the install folder (e.g. `C:\Program Files\7-Zip`) to your `PATH` so the `7z` command works from the terminal.

It makes sense to decide on one person to actually run and submit the script. **Later submissions overwrite earlier ones.** Also think about how you will schedule future runs, as new challenges might run throughout the entire tournament.

## 🚀 OOOOOKAY, let's goo

To check upcoming games, your progress and the leaderboard, visit **https://c2f.public.quantco.cloud/leaderboard/**.
