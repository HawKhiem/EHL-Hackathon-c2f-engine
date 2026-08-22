# Claim to Fame shortcuts.  Usage:
#   make 7             play game 7 (start it a few seconds before the game opens)
#   make play G=7      same thing
#   make check         real model against the permanent test game 0 (needs an LLM key in .env)
#   make mock          pipeline test against game 0 with a canned model answer
#   make test          unit tests
#   make learn G=7     after game 7 closes: infer t bounds + recalibrate b_scale (commits+pushes)
#   make truth G=7      infer fair-value bounds for finished game 7 -> runs/truth_game_07.json
# After make N / make play, the run log runs/game_NN.json is committed and pushed
# (set PUSH=0 to skip). A failed run still commits its log. The truth inference is then
# attempted too and runs/truth_game_NN.json committed if it succeeded (it needs the game
# to show as completed on the leaderboard; rerun `make truth G=N` later otherwise).
PY := pixi run python
PUSH ?= 1

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
define push_truth
	-$(PY) -m c2f.truth $(1)
	$(call push_file,truth_,$(1),truth)
endef

GAMES := $(shell seq 0 100)
.PHONY: play check mock test fb truth $(GAMES)

$(GAMES):
	-$(PY) -m c2f.run $@
	$(call push_log,$@)
	$(call push_truth,$@)

play:
	@test -n "$(G)" || { echo "usage: make play G=<game_id>"; exit 1; }
	-$(PY) -m c2f.run $(G)
	$(call push_log,$(G))
	$(call push_truth,$(G))

check:
	$(PY) -m c2f.run 0

mock:
	$(PY) -m c2f.run 0 --mock

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
	$(call push_truth,$(G))
