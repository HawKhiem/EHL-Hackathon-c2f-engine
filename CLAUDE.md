# CLAUDE.md

See **[AGENTS.md](AGENTS.md)** — single source of truth for architecture, commands,
conventions, and the traps in this codebase. Read it before making changes.

**The challenge solution lives in `c2f/`.** Its architecture, input structure, pricing rule
and commands are in **[docs/C2F-ARCHITECTURE.md](docs/C2F-ARCHITECTURE.md)** — read that
before touching `c2f/`, `tests/`, or the `Makefile`. Design spec:
`docs/superpowers/specs/2026-08-22-c2f-engine-design.md`.

Quick links: [README.md](README.md) (quickstart) · [GAME_DESCRIPTION.md](GAME_DESCRIPTION.md)
(the brief) · [API_HANDBOOK.md](API_HANDBOOK.md) · [docs/AGENT-TOOLING.md](docs/AGENT-TOOLING.md)

**This team runs on OpenAI only.** `c2f/` has no Anthropic code path — `OPENAI_API_KEY` is
the only key `c2f/llm.py` and `c2f/policy.py` look for. Do not reintroduce Anthropic/Claude
model calls.
