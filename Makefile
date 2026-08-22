# Claim to Fame shortcuts.  Usage:
#   make 7             play game 7 (start it a few seconds before the game opens)
#   make play G=7      same thing
#   make check         real model against the permanent test game 0 (needs an LLM key in .env)
#   make mock          pipeline test against game 0 with a canned model answer
#   make test          unit tests
PY := pixi run python

GAMES := $(shell seq 0 100)
.PHONY: play check mock test $(GAMES)

$(GAMES):
	$(PY) -m c2f.run $@

play:
	@test -n "$(G)" || { echo "usage: make play G=<game_id>"; exit 1; }
	$(PY) -m c2f.run $(G)

check:
	$(PY) -m c2f.run 0

mock:
	$(PY) -m c2f.run 0 --mock

test:
	$(PY) -m pytest -q tests
