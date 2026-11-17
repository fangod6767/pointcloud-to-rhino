# PointCloud to Rhino · 建筑点云分割与网格重建

把彩色室内扫描点云转换为可审阅的语义分组，并生成能在 Rhino 中继续编辑的网格。

建筑学课程实践项目，作者：樊奕翔。AI 协助代码实现、测试与实验整理，项目取舍和局限详见 Notebook。

![分组与网格预览](docs/images/mesh_preview.png)

## 工作流程

**彩色点云 → RandLA-Net 语义分割 → RANSAC 平面提取 + DBSCAN 空间聚类 → 人工审阅 → 贴合点云的网格 → 3DM**

- 独立桌面窗口，支持导入扫描点云、查看语义与候选组、调整分组和导出模型。
- 13 类室内语义，包括墙、地板、天花、门、窗、桌椅等。
- Ball Pivoting 生成贴合实测点的网格，3DM 图层包含物体类别和组号。
- 随机森林是单独训练的对照基线，不参与客户端主流程。
- Notebook 包含方法解释、源码、历史实验结果和抽样测试入口。

输出是 **Mesh**，不保证封闭，也不是自动生成的 BIM 或 NURBS 实体。几何分组是候选组，仍需人工判断。该预训练模型面向室内 S3DIS 数据；室外、无颜色或不同扫描条件下的质量需要重新验证。

## 安装与启动（Windows）

本项目原环境使用 Python 3.12。以下命令在仓库根目录的 PowerShell 中执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe download_weights.py
.\.venv\Scripts\python.exe pointcloud_client/app.py
```

依赖较大，首次安装需要联网。版本清单来自原项目环境；本次发布检查复用了该环境，尚未在全新机器上验证安装。默认客户端使用 CPU 推理，不强制要求独立显卡。源码启动不需要安装 Rhino；查看、继续编辑导出的 3DM 可使用 Rhino。

在窗口中导入彩色 PCD/PLY 等扫描文件，确认单位换算与竖直轴，运行分割与建模，审阅候选组后导出 3DM。可调整网格采样间距和最大连边长度。没有 RGB 时的中性颜色回退未经准确率验证。

## Notebook 与现场抽样

打开 [课程展示 Notebook](建筑点云分割与网格重建_课程展示.ipynb)：

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

公开版保留原课程实验的历史输出、图表和统计证据，移除了学号、本机用户路径和内嵌原始点云样本。**打开文件看到的输出不代表你刚刚运行的结果。** 运行全部单元格时，缺少外部数据、权重或 3DM 的环节会明确跳过；历史统计的复算仍可执行。原始课堂提交版未被修改。

准备默认 `office_2` 房间的真实数据和预测：

```powershell
.\.venv\Scripts\python.exe prepare_notebook_sample.py
```

该步骤从第三方镜像读取一个真实 S3DIS 房间，下载权重并实际推理，CPU 上可能需要数分钟或更久。请先阅读下方数据来源及原始使用条款。已有输出时脚本会停止，避免覆盖实验。镜像若不再支持 Range 请求，脚本会报错，需从原始来源取得数据。

随后在 Notebook 按顺序运行初始化，再运行 5.1：修改 `CHECK_SEED` 和 `CHECK_POINTS`，可抽取不同点并比较真实类、预测类、准确率和 mIoU。默认 `CHECK_ROOM='office_2'`；其他房间须另行准备对应数据与预测。这是对预测结果的抽样检查，**不会重新训练模型**。

5.2 需要将 `LIVE_INPUT` 改为自己的扫描路径；没有对应真实标签时设 `LIVE_TRUTH=None`。此时可以实际推理、观察结果，但不能计算有监督准确率。5.3 中原课程 3DM 回读依赖未随仓库分发的历史文件。

## 实验结果与边界

| 历史实验 | 测试范围 | OA | mIoU |
|---|---|---:|---:|
| 随机森林 | Area 3 / office_1 | 48.31% | 29.95% |
| 预训练 RandLA-Net | Area 3 / office_1 | 93.24% | 81.37% |
| 预训练 RandLA-Net 固定批次 | Area 3 全部 23 个房间 | 87.61% | 70.20% |

随机森林只在 Area 1 / office_1 训练，RandLA-Net 使用外部预训练权重，**未在本项目训练或微调**。两者训练数据规模不同，因此不能把差异完全归因于算法结构。本项目不是完整六折交叉验证，也未独立审计权重的全部上游训练过程。

表中结果来自已保存的历史流程；客户端加入局部坐标原点处理后，体素划分与结果可能变化，不能把这些成绩直接作为当前客户端的准确率保证。报告保存在 [docs/evidence](docs/evidence)，Notebook 解释了这些差异。

## 数据、权重与引用

- S3DIS：Armeni 等，*3D Semantic Parsing of Large-Scale Indoor Spaces*, CVPR 2016。[原项目与数据入口](http://buildingparser.stanford.edu/dataset.html)。下载工具使用 [第三方镜像](https://huggingface.co/datasets/cminst/S3DIS)，镜像标签不能替代原始数据条款。
- Hu 等，*RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds*, CVPR 2020。[论文](https://arxiv.org/abs/1911.11236)。
- 网络实现改编自 [Open3D-ML](https://github.com/isl-org/Open3D-ML)，保留其 [MIT 声明](vendor/open3d_ml/LICENSE)。固定上游提交、权重来源与 SHA-256 见 [weights/provenance.json](weights/provenance.json)。
- `download_weights.py` 下载约 60 MB 的官方发布权重并校验 SHA-256；权重不存入 Git。

仓库不附完整数据集、原始点云样本、模型权重、安装环境、打包 EXE 或历史 3DM。使用者自行取得所需资产并遵守各自条款。本仓库目前未对原创代码声明统一开源许可证；公开展示不改变第三方文件已有许可。

## 文件入口与验证

| 文件 | 用途 |
|---|---|
| `segmentation_pipeline.py` | 点云预处理、模型推理、语义输出 |
| `randlanet_portable.py` | 适配后的 RandLA-Net 网络 |
| `semantic_groups.py` | 语义内 RANSAC / DBSCAN 分组 |
| `comparison_baseline.py` | 随机森林训练与评估 |
| `pointcloud_client/app.py` | 桌面客户端 |
| `pointcloud_client/mesh_reconstruction.py` | 网格重建 |
| `pointcloud_client/exporter.py` | 带中文类别图层的 3DM 导出 |

无需真实数据的读取与网格测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s pointcloud_client -p test_scan.py -v
.\.venv\Scripts\python.exe -m unittest discover -s pointcloud_client -p test_mesh.py -v
```

其他测试包含对本地真实样例、GUI 或历史成果的依赖，不能在缺少这些文件时直接当作通用测试套件运行。
