# SPDX-FileCopyrightText: 2025 mohamed hamdi <haamdi@outlook.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY: configure build install run clean all

PREFIX ?= ~/.local
BINARY  := audconv

all: configure build

configure: pyproject.toml uv.lock
	@mise use python@3.12
	@uv sync
	@echo "Configured!"

generate-completions:
	@mkdir -p dist
	@PATH="$(PWD):$(PATH)" _AUDCONV_COMPLETE=bash_source $(BINARY) > dist/$(BINARY).bash
	@PATH="$(PWD):$(PATH)" _AUDCONV_COMPLETE=zsh_source $(BINARY) > dist/_$(BINARY).zsh
	@echo "Generated shell completions"

build: main.py
	@uv run pyinstaller --onefile main.py --log-level=FATAL
	@cp dist/main $(BINARY)
	@$(MAKE) generate-completions
	@echo "Built successfully!"

install: build
	@install -Dm755 $(BINARY) $(PREFIX)/bin/$(BINARY)
	@install -Dm644 dist/$(BINARY).bash \
	            $(PREFIX)/share/bash-completion/completions/$(BINARY)
	@install -Dm644 dist/_$(BINARY).zsh \
	            ~/.zsh/completions/_$(BINARY)
	@echo "Installed!"

run: build
	@uv run $(BINARY)

clean:
	@rm -rf $(BINARY) dist build main.spec
	@rm -rf .venv __pycache__ *.pyc
	@echo "Cleaned!"
