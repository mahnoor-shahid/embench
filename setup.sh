#!/usr/bin/env bash
# One-command dev setup for macOS / Linux.
#   ./setup.sh            # core + dev (pytest)
#   ./setup.sh all        # every model backend too
#   ./setup.sh openai,google
#
# Creates a local .venv, installs embench editable with the chosen extras,
# and seeds a .env from .env.example if you don't have one yet.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
EXTRAS="${1:-dev}"

echo "Creating virtual environment in .venv ..."
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# always include dev so tests run; merge with any user-requested extras
case ",$EXTRAS," in
  *,dev,*) ;;
  *) EXTRAS="$EXTRAS,dev" ;;
esac
echo "Installing embench [$EXTRAS] ..."
.venv/bin/python -m pip install -e ".[$EXTRAS]"

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
