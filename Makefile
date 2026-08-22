PYTHON := pixi run python
GAME ?= 0

.PHONY: setup history estimate submit

setup:
	pixi install

# Rebuild runs/history.json from new truth files. Run this BETWEEN rounds, never inside the
# 60s submission window - it is deliberately not a prerequisite of estimate/submit.
history:
	$(PYTHON) -m c2f.history

estimate:
	$(PYTHON) -m c2f.run --game $(GAME) --no-submit

# Start this BEFORE the round begins: it polls for the key, then decrypts, estimates and
# submits the moment the game opens.
submit:
	$(PYTHON) -m c2f.run --game $(GAME)
