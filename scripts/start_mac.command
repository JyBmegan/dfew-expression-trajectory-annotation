#!/bin/zsh
set -e
cd "$(dirname "$0")/.."
echo "正在启动正式标注，请不要关闭这个窗口……"
if [ ! -d .venv ]; then
  echo "第一次启动：正在安装本地运行环境。"
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi
if [ ! -f config/project.toml ]; then
  echo "没有找到个人启动包中的 config/project.toml。"
  echo "请重新解压负责人发给你的个人启动包，并选择合并到本仓库。"
  read "?按任意键关闭……"
  exit 1
fi
if [ ! -f local_data/study.sqlite ]; then
  echo "没有找到个人任务文件 local_data/study.sqlite。"
  echo "请重新解压负责人发给你的个人启动包。"
  read "?按任意键关闭……"
  exit 1
fi
python manage.py annotator-check --config config/project.toml --database local_data/study.sqlite
python manage.py run --config config/project.toml
