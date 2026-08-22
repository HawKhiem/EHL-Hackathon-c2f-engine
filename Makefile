# Claim to Fame shortcuts.  Usage:
#   make 7             play game 7 (start it a few seconds before the game opens)
#   make play G=7      same thing
#   make check         real model against the permanent test game 0 (needs an LLM key in .env)
#   make test          unit tests
#   make learn G=7     after game 7 closes: infer t bounds + recalibrate bias/sigma/acceptance (commits+pushes)
#   make truth G=7      infer fair-value bounds for finished game 7 -> runs/truth_game_07.json
# After make N / make play, the run log runs/game_NN.json is committed and pushed
# (set PUSH=0 to skip). A failed run still commits its log. The truth inference is then
# attempted too and runs/truth_game_NN.json committed if it succeeded (it needs the game
# to show as completed on the leaderboard; rerun `make truth G=N` later otherwise).
# When a truth file exists, the calibration is refit and runs/calibration.json committed
# if it changed -- so `make learn` is only needed to redo this by hand.
PY := pixi run python
PUSH ?= 1
# after a game, wait this long for it to close on the leaderboard before inferring truth
TRUTH_WAIT ?= 60

# $(call push_file,<prefix>,<game>,<msg>) commits+pushes runs/<prefix>_game_NN.json if present.
define push_file
	@if [ "$(PUSH)" = "1" ] && [ -f runs/$(1)game_$$(printf '%02d' $(2)).json ]; then \
	  f=runs/$(1)game_$$(printf '%02d' $(2)).json; \
	  git add "$$f" && git commit -q -m "$(3): game $(2)" -- "$$f" && git push -q && echo "pushed $$f" \
	  || echo "warning: could not commit/push $$f"; \
	fi
endef
define push_log
	$(call push_file,,$(1),run log)
endef
# $(call push_truth,<game>,<wait seconds>) waits for the game to close, then infers t bounds
# and refits the calibration from every truth + feedback file we have.
define push_truth
	@if [ "$(2)" -gt 0 ]; then \
	  echo "waiting $(2)s for game $(1) to close on the leaderboard..."; sleep $(2); \
	fi
	-$(PY) -m c2f.truth $(1)
	$(call push_file,truth_,$(1),truth)
	$(call push_calibration,$(1))
endef

# $(call push_calibration,<game>) recalibrates once game <game>'s truth file exists.
define push_calibration
	@if [ -f runs/truth_game_$$(printf '%02d' $(1)).json ]; then \
	  $(PY) -m c2f.calibrate || echo "warning: calibration failed"; \
	  if [ "$(PUSH)" = "1" ] && [ -n "$$(git status --porcelain -- runs/calibration.json)" ]; then \
	    git add runs/calibration.json \
	    && git commit -q -m "calibration: after game $(1)" -- runs/calibration.json \
	    && git push -q && echo "pushed runs/calibration.json" \
	    || echo "warning: could not commit/push runs/calibration.json"; \
	  fi; \
	fi
endef

GAMES := $(shell seq 0 100)
.PHONY: play check test fb truth backtest rescore hooks $(GAMES)

$(GAMES):
	-$(PY) -m c2f.run $@
	$(call push_log,$@)
	$(call push_truth,$@,$(TRUTH_WAIT))

play:
	@test -n "$(G)" || { echo "usage: make play G=<game_id>"; exit 1; }
	-$(PY) -m c2f.run $(G)
	$(call push_log,$(G))
	$(call push_truth,$(G),$(TRUTH_WAIT))

check:
	$(PY) -m c2f.run 0

test:
	$(PY) -m pytest -q tests

learn:
	@test -n "$(G)" || { echo "usage: make learn G=<game_id>"; exit 1; }
	$(PY) -m c2f.truth $(G)
	$(PY) -m c2f.calibrate
	@if [ "$(PUSH)" = "1" ]; then \
	  git add runs/truth_game_$$(printf '%02d' $(G)).json runs/calibration.json && \
	  git commit -q -m "learn: game $(G) truth + calibration" -- runs/truth_game_$$(printf '%02d' $(G)).json runs/calibration.json && git push -q && echo "pushed" \
	  || echo "warning: could not commit/push"; \
	fi

# Digest a finished game from the public leaderboard: inferred t bounds, our verdicts
#   make fb G=2
fb:
	@test -n "$(G)" || { echo "usage: make fb G=<game_id>"; exit 1; }
	$(PY) -m c2f.feedback $(G)

truth:
	@test -n "$(G)" || { echo "usage: make truth G=<game_id>"; exit 1; }
	$(call push_truth,$(G),0)

# THE evaluation: replay the CURRENT strategy on past games against the real opponents
# (uses feedback + truth + calibration as components; calls the model, ~40 s/game)
#   make backtest            all completed games decrypted locally
#   make backtest G="2 4 6"  specific games
backtest:
	$(PY) -m c2f.backtest $(G)

# Re-score the stored replays without calling the model (after changing price.py only)
rescore:
	$(PY) -m c2f.backtest --no-llm $(G)

# Install the git hook that blocks algorithm commits without a fresh backtest
hooks:
	git config core.hooksPath .githooks
	@echo "hooks installed (core.hooksPath=.githooks)"
