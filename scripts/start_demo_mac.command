#!/bin/zsh
set -e
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi
python manage.py make-demo
if [ ! -f local_data/demo.sqlite ]; then
  python manage.py init-db --database local_data/demo.sqlite
  python manage.py prepare-study --config demo_data/project.toml --database local_data/demo.sqlite --selection-output local_data/demo_training_selection.csv
fi
python manage.py run --config demo_data/project.toml --database local_data/demo.sqlite
