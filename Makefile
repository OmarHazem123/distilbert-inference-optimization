# Thin wrapper around tasks.py -- the single source of truth for every task.
# On Windows without `make`, run `python tasks.py <target>` instead.

PYTHON ?= python
TASKS  := setup doctor predict export quantize benchmark benchmark-dynamic profile \
          evaluate report all \
          lint fmt typecheck test test-fast check clean

.PHONY: help $(TASKS)

help:
	@$(PYTHON) tasks.py --list

$(TASKS):
	@$(PYTHON) tasks.py $@
