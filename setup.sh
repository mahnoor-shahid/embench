#!/usr/bin/env bash
# One-command dev setup for macOS / Linux.
#   ./setup.sh
#
# Creates a local .venv, installs embench editable (API backends + dev tools),
# and seeds a .env from .env.example if you don't have one yet. To also work on
# the local backend (PyTorch), run afterwards:  pip install -e ".[all]"
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "Creating virtual environment in .venv ..."
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

echo "Installing embench (core + API backends + dev tools) ..."
.venv/bin/python -m pip install -e ".[dev]"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example - add your API keys."
fi

cat <<'EOF'

Done. Next:
  source .venv/bin/activate     # activate the environment
  pytest                         # run the tests
  embench run -m dummy:256 --retrieval sample_data/retrieval.json
EOF
