@echo off
setlocal
cd /d "%~dp0\.."
if not exist .venv (
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -r requirements.txt
)
call .venv\Scripts\activate.bat
if not exist config\project.toml (
  copy config\annotator.example.toml config\project.toml
  echo Edit config\project.toml to point to the authorized local DFEW files, then start again.
  pause
  exit /b 1
)
python manage.py run --config config/project.toml
