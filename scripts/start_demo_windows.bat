@echo off
setlocal
cd /d "%~dp0\.."
if not exist .venv (
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python manage.py make-demo
if not exist local_data\demo.sqlite (
  python manage.py init-db --database local_data/demo.sqlite
  python manage.py prepare-study --config demo_data/project.toml --database local_data/demo.sqlite --selection-output local_data/demo_training_selection.csv
)
python manage.py run --config demo_data/project.toml --database local_data/demo.sqlite
