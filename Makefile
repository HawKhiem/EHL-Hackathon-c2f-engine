# Claim to Fame shortcuts.  Usage:
#   make 7             play game 7 (start any time in the 15 min before it opens)
#   make play G=7      same thing  (safe to start early: KEY_WAIT_S=900 by default)
#   make check         real model against the permanent test game 0 (needs an LLM key in .env)
#   make test          unit tests
#   make autoplay      play EVERY upcoming round automatically with v2 (Ctrl-C to stop)
#   make learn G=7     after game 7 closes: infer t bounds + recalibrate bias/sigma/acceptance (commits+pushes)
#   make truth G=7      infer fair-value bounds for finished game 7 -> runs/truth_game_07.json
#   make history        the t ranges the market accepted, as shown to the model -> runs/market_history.txt
#   make deviation      how far t/a/b sit from the market's proven fair value, and what it cost
#   make postmortem     attribute past rounds' losses to named causes, ranked by euros
#   make autotune       propose constant changes from that, gated on the backtest (no writes)
#   make propose        ask a model for a PROMPT rule from the same evidence (no writes)
#   make stage          stage that rule, then `make replay` scores it before anyone edits SYSTEM
# After make N / make play, the run log runs/game_NN.json is committed and pushed
# (set PUSH=0 to skip). A failed run still commits its log. The truth inference is then
# attempted too and runs/truth_game_NN.json committed if it succeeded (it needs the game
# to show as completed on the leaderboard; rerun `make truth G=N` later otherwise).
# When a truth file exists, the calibration is refit AND the market-history block the prompt
# shows the model is regenerated; runs/calibration.json and runs/market_history.txt are
# committed if they changed -- so `make learn` is only needed to redo this by hand.
PY := pixi run python
PUSH ?= 1
# How long get_case.sh polls for the decryption key before giving up. run.py restarts the
# 60 s clock the moment the key appears, so waiting longer costs nothing and `make N` can be
# started well before the game opens. Game 25 was lost to the old 120 s budget plus a curl
# blip that killed the poll outright.
export KEY_WAIT_S ?= 900
# after a game, wait this long for it to close on the leaderboard before inferring truth
TRUTH_WAIT ?= 60

# $(call push_file,<prefix>,<game>,<msg>) commits+pushes runs/<prefix>_game_NN.json if present.
define push_file
	@echo "kept locally, not committed (runs/ is claim data and gitignored): runs/$(1)game_$$(printf '%02d' $(2)).json"
endef
# Run logs embed the case's policy/description/invoice text = CLAIM DATA, which must never be
# checked in (hackathon rule). They stay on disk (the pricing engine's memory reads them) and are
# gitignored; only the truth bounds and the calibration - numbers - are committed after a round.
define push_log
	@echo "run log kept locally (claim data is not committed): runs/game_$$(printf '%02d' $(1)).json"
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

# $(call push_calibration,<game>) recalibrates once game <game>'s truth file exists, and
# refreshes the market-history snapshot the prompt shows the model (c2f.history).
define push_calibration
	@if [ -f runs/truth_game_$$(printf '%02d' $(1)).json ]; then \
	  $(PY) -m c2f.calibrate || echo "warning: calibration failed"; \
	  $(PY) -m c2f.history >/dev/null || echo "warning: market history refresh failed"; \
	  if [ "$(PUSH)" = "1" ] && [ -n "$$(git status --porcelain -- runs/calibration.json runs/market_history.txt)" ]; then \
	    git add runs/calibration.json runs/market_history.txt \
	    && git commit -q -m "calibration: after game $(1)" -- runs/calibration.json runs/market_history.txt \
	    && git push -q && echo "pushed runs/calibration.json runs/market_history.txt" \
	    || echo "warning: could not commit/push runs/calibration.json"; \
	  fi; \
	fi
endef

GAMES := $(shell seq 0 100)
.PHONY: play check test fb truth history backtest replay rescore deviation postmortem autotune tune propose stage unstage $(GAMES)

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

autoplay:
	$(PY) -m c2f.autoplay --strategy v2

test:
	$(PY) -m pytest -q tests

learn:
	@test -n "$(G)" || { echo "usage: make learn G=<game_id>"; exit 1; }
	$(PY) -m c2f.truth $(G)
	$(PY) -m c2f.calibrate
	$(PY) -m c2f.history >/dev/null
	@if [ "$(PUSH)" = "1" ]; then \
	  git add runs/truth_game_$$(printf '%02d' $(G)).json runs/calibration.json runs/market_history.txt && \
	  git commit -q -m "learn: game $(G) truth + calibration" -- runs/truth_game_$$(printf '%02d' $(G)).json runs/calibration.json runs/market_history.txt && git push -q && echo "pushed" \
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

# The t ranges the market actually accepted, per category and per repeated line label, as the
# block c2f/llm.py puts in front of every prompt. Refreshed automatically after every round
# (push_calibration); run it by hand to see the block, or after back-filling old truth files.
history:
	$(PY) -m c2f.history

# THE evaluation: replay the CURRENT strategy on past games against the real opponents
# (uses feedback + truth + calibration as components; calls the model, ~40 s/game)
#   make backtest            re-price + re-score the stored estimates of the last 5 completed decrypted
#                            games with the CURRENT price.py + calibration - no model call, seconds
#   make replay              call the model again for every old game, then score (prompt/extract changes)
#   make replay G="2 4"      call the model for these games only; the others come from the store
#   (the verdict is always over the LAST 5 completed decrypted games)
backtest:
	$(PY) -m c2f.backtest $(G)

replay:
	$(PY) -m c2f.backtest --replay $(G)

# old name for the default behaviour
rescore: backtest

# How far t, a and b sit from the fair value the market proved, per bucket and per game,
# plus the euros each mistake cost against an oracle that knew t.
#   make deviation                       the boards we actually submitted
#   make deviation REPRICE=1             stored estimates, priced by today's constants
#   make deviation G=14-24               only those games
#   make deviation SWEEP=B_QUANTILE=0.27,0.3333,0.42   one line per candidate value
deviation:
	$(PY) -m c2f.deviation $(if $(G),--games $(G),) $(if $(REPRICE),--reprice,) $(if $(SWEEP),--sweep $(SWEEP),)

# Attribute a finished round's money to named causes, with the action for each.
#   make postmortem G=10     one game
#   make postmortem          every game with a run log + truth file
postmortem:
	$(PY) -m c2f.postmortem $(if $(G),$(G),--all)

# Propose constant changes from the post-mortem, and accept one only if it passes all
# three gates: total improves, a majority of individual games improve, and
# `c2f.backtest` still reports SUCCESS. Proposal only by default.
#   make autotune            report what would change and why, touch nothing
#   make tune                same, then write runs/tuning.json for whatever passed
autotune:
	$(PY) -m c2f.autotune

tune:
	$(PY) -m c2f.autotune --apply

# The other half of the loop: the causes no constant can reach (coverage, abstention).
# Asks a model for a prompt rule from the post-mortem's own evidence, shows it the
# CURRENT prompt so it cannot restate what is already there, and writes nothing.
#   make propose      print the evidence and the proposed rules
#   make stage        stage the top rule in runs/prompt_addendum.txt, then: make replay
#   make unstage      drop the staged rule
propose:
	$(PY) -m c2f.propose

stage:
	$(PY) -m c2f.propose --write

unstage:
	$(PY) -m c2f.propose --clear

