# Claim to Fame shortcuts.  Usage:
#   make 7             play game 7 (start it a few seconds before the game opens)
#   make play G=7      same thing
#   make check         real model against the permanent test game 0 (needs an LLM key in .env)
#   make mock          pipeline test against game 0 with a canned model answer
#   make test          unit tests
# After make N / make play, the run log runs/game_NN.json is committed and pushed
# (set PUSH=0 to skip). A failed run still commits its log.
PY := pixi run python
PUSH ?= 1

define push_log
	@if [ "$(PUSH)" = "1" ] && [ -f runs/game_$$(printf '%02d' $(1)).json ]; then \
	  f=runs/game_$$(printf '%02d' $(1)).json; \
	  git add "$$f" && git commit -q -m "run log: game $(1)" -- "$$f" && git push -q && echo "pushed $$f" \
	  || echo "warning: could not commit/push $$f"; \
	fi
endef

GAMES := $(shell seq 0 100)
.PHONY: play check mock test $(GAMES)

$(GAMES):
	-$(PY) -m c2f.run $@
	$(call push_log,$@)

play:
	@test -n "$(G)" || { echo "usage: make play G=<game_id>"; exit 1; }
	-$(PY) -m c2f.run $(G)
	$(call push_log,$(G))

check:
	$(PY) -m c2f.run 0

mock:
	$(PY) -m c2f.run 0 --mock

test:
	$(PY) -m pytest -q tests
