# DFEW 表情轨迹标注平台

这是一个可在 macOS 和 Windows 本地运行的浏览器标注平台。标注者只会看到模型实际接收的 DFEW 16 帧人脸序列，不会看到数据集标签、模型预测或其他标注者的答案。

当前 GitHub 仓库只提供标注平台、中文说明和虚构人脸 Demo。后续分析代码保留在协调者的本地工作区，不放入标注者需要下载的仓库。

## 先看 Demo

Demo 不包含 DFEW 图像：

- macOS：双击 `scripts/start_demo_mac.command`
- Windows：双击 `scripts\\start_demo_windows.bat`

也可以在仓库目录运行：

```bash
python manage.py make-demo
python manage.py init-db --database local_data/demo.sqlite
python manage.py prepare-study --config demo_data/project.toml --database local_data/demo.sqlite --selection-output local_data/demo_training_selection.csv
python manage.py run --config demo_data/project.toml --database local_data/demo.sqlite
```

浏览器打开 `http://127.0.0.1:5050`，Demo 测试账号为 `R01`。Demo 账号只存在于本机的合成数据库中，不是正式研究账号。

## 正式数据放置位置

推荐将 GitHub 仓库和私有数据放在同一级目录：

```text
/your/workspace/
├── trajectory_sampling_study/       # GitHub clone
└── dfew_private/                    # 私有数据，不上传 GitHub
    ├── clip_224x224_16f.zip         # 官方 16 帧压缩包
    ├── full_length_archives/        # 可选：原始长度档案
    ├── train_set_1.csv
    ├── test_set_1.csv
    ├── annotation.xlsx              # DFEW 十人投票表
    └── checkpoints/
        ├── alexnet.pth
        └── resnet18.pth
```

数据也可以继续留在你现在的本地目录，不需要搬动。程序支持加密压缩包，也支持解压目录；解压目录应包含类似 `00001/1.jpg` 到 `00001/16.jpg` 的文件。

## Google Drive 下载包

给标注者下载的 Google Drive 包只需要包含官方的 `clip_224x224_16f.zip`。建议保持官方压缩包原样上传，不要把几万个小图片文件拆开上传。GitHub 页面可以放置链接：

```text
DFEW 16 帧数据下载：由协调者在这里粘贴 Google Drive 链接
```

下载后将压缩包放在 `local_data/dfew/clip_224x224_16f.zip`，然后运行自动配置脚本。压缩包密码请通过单独的私密渠道发送，不要写在 GitHub 或 Google Drive 公开说明中。`train_set_1.csv`、`test_set_1.csv`、`annotation.xlsx`、完整长度档案、模型检查点和 `study.sqlite` 不属于标注者下载包，由协调者私下管理。

## 自动生成本地配置

最简单的方式是把私有文件放到仓库的 `local_data/dfew/` 下，然后运行：

```bash
python scripts/setup_local_config.py
```

脚本会检查必需文件，询问压缩包密码，并自动生成被 Git 忽略的 `config/project.toml`。如果数据已经在其他位置，直接指定已有目录：

```bash
python scripts/setup_local_config.py --dfew-root "/path/to/your/dfew_private"
```

如果文件名或位置不同，可额外使用 `--archive`、`--annotation`、`--train-csv`、`--test-csv`、`--alexnet` 和 `--resnet18`。密码只写入本地的 `config/project.toml`，不会进入 GitHub。

协调者完成配置后运行：

```bash
python manage.py init-db
python manage.py prepare-study --config config/project.toml
python manage.py run --config config/project.toml
```

## 三个正式账号的角色

正式研究固定使用三个代码：

- `R01`：独立标注者 1
- `R02`：独立标注者 2
- `R03`：裁决者，只接收触发分歧规则的任务

R01 和 R02 对正式单帧任务及 16 帧序列独立评分。只有当类别不同、强度曲线最大差达到 3、峰值位置差至少 4 帧，或曲线形状不一致时，系统才为 R03 生成裁决任务。R03 不参与正式主队列的独立评分。三人可以共同完成校准任务，用于统一尺度理解。

## 随机顺序与断点保存

单帧任务使用固定随机种子生成可复现的乱序队列，并保证同一视频在同一标注者队列中至少间隔 50 个任务。若已生成刺激特征，系统还会在可行时避开相同视觉指纹的相邻任务。随机化不能证明任意两张“看起来相似”的图片绝不会相邻，因此正式分析会保留视频 ID 和队列位置用于检查；同一视频的相邻帧不会连续出现。

打开一个任务后，页面会立即将任务标记为进行中。操作过程中的序列曲线和勾选项会自动保存为草稿；关闭浏览器或电脑后重新登录，会回到第一个未完成任务。点击“保存并继续”后才会提交正式答案，已提交任务不会重复出现。

## 标注者看到什么表示完成

右上角的“进度”页面会显示单帧和序列任务的完成数。某一账号的所有进度条达到 100%，并看到“全部任务已完成”页面，就表示该账号完成。协调者还应收到该账号导出的文件夹，而不是只根据浏览器页面判断。

## 标注者完成后需要交回什么

标注者关闭浏览器后，在仓库目录运行（将代码替换为自己的账号）：

```bash
python manage.py export --annotator R01 --output exports/R01
```

需要交回整个 `exports/R01/` 文件夹。它包含任务标识、类别、强度曲线、可见情况、完成时间和版本信息，不包含 DFEW 图像。不要交回 `config/project.toml`，因为其中可能有压缩包密码。

协调者收齐 `exports/R01`、`exports/R02` 后运行：

```bash
python manage.py merge exports/R01 exports/R02
python manage.py create-adjudications
```

如果生成了裁决任务，再为 R03 创建私有裁决包并导入；最后导出共识结果：

```bash
python manage.py export-consensus --output exports/consensus
python manage.py validate-study --require-complete
```

## 数据隐私

DFEW 图像、数据库、标注结果、模型检查点、压缩包密码和本机绝对路径均被 Git 忽略。公开仓库只包含代码、文档和合成 Demo。DFEW 数据集请按 Jiang 等人的原始论文引用。
