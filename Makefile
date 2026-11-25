# Thin wrapper around tasks.py -- the single source of truth for every task.
# On Windows without `make`, run `python tasks.py <target>` instead.

PYTHON ?= python
TASKS  := setup doctor predict export lint fmt typecheck test check clean

.PHONY: help $(TASKS)

help:
	@$(PYTHON) tasks.py --list

$(TASKS):
	@$(PYTHON) tasks.py $@
