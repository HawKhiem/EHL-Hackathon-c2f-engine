PYTHON := python3
GAME ?= 0

.PHONY: setup history estimate submit

setup:
	$(PYTHON) -m pip install --quiet requests openai pypdf
	@command -v 7z >/dev/null 2>&1 || brew install p7zip

history:
	$(PYTHON) -m c2f.history

estimate: history
	$(PYTHON) -m c2f.run --game $(GAME) --no-submit

submit: history
	$(PYTHON) -m c2f.run --game $(GAME)
