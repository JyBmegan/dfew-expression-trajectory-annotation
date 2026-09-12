# 标注者启动说明

## 你会收到什么

协调者会单独提供：

1. 标注者自己的 `study.sqlite` 私有数据库；
2. DFEW 16 帧压缩包及授权访问方式；
3. 你的标注代码（正式任务使用 `R01` 或 `R02`，裁决任务使用 `R03`）。

公开 GitHub 仓库不包含 DFEW 图像、数据库或密码。

## 第一次启动

安装 Python 3.11 或 3.12，将私有数据库放到仓库的 `local_data/study.sqlite`，复制 `config/annotator.example.toml` 为 `config/project.toml`，填入本机压缩包路径和密码，然后：

- macOS：双击 `scripts/start_mac.command`；
- Windows：双击 `scripts/start_windows.bat`。

浏览器会打开本机页面。输入协调者提供的代码。正式队列开始前会先显示校准任务。

## 标注过程中

- 可以关闭浏览器或电脑，已完成答案和未提交草稿都会保留。
- 重新登录后会回到第一个未完成任务。
- 不要修改正在使用的数据库文件名。
- 正式任务中不要与其他标注者讨论具体片段。

## 完成标志

进入“进度”页面，看到所有任务进度为 100%，并且首页不再显示待完成任务，即表示浏览器端已完成。之后仍需导出结果交给协调者。

## 交回文件

关闭浏览器后，在仓库目录运行（将代码替换为你的实际代码）：

```bash
python manage.py export --annotator R01 --output exports/R01
```

将整个 `exports/R01/` 文件夹交给协调者。不要交回 `config/project.toml`，因为其中可能包含压缩包密码，也不要交回 DFEW 图像。
