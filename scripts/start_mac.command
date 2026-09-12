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
if [ ! -f config/project.toml ]; then
  cp config/annotator.example.toml config/project.toml
  echo "Edit config/project.toml to point to the authorized local DFEW files, then start again."
  read -k 1
  exit 1
fi
python manage.py run --config config/project.toml
