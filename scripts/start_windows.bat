@echo off
setlocal
cd /d "%~dp0\.."
echo 正在启动正式标注，请不要关闭这个窗口……
if not exist .venv (
  echo 第一次启动：正在安装本地运行环境。
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -r requirements.txt
)
call .venv\Scripts\activate.bat
if not exist config\project.toml (
  echo 没有找到个人启动包中的 config\project.toml。
  echo 请重新解压负责人发给你的个人启动包，并选择合并到本仓库。
  pause
  exit /b 1
)
if not exist local_data\study.sqlite (
  echo 没有找到个人任务文件 local_data\study.sqlite。
  echo 请重新解压负责人发给你的个人启动包。
  pause
  exit /b 1
)
python manage.py annotator-check --config config/project.toml --database local_data/study.sqlite
if errorlevel 1 pause & exit /b 1
python manage.py run --config config/project.toml
