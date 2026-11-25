# PointCloud Desk v0.4.0 · Windows 便携客户端

下载 Assets 中的 **PointCloudDesk-Windows-x64.zip**，完整解压后双击 `PointCloudDesk/PointCloudDesk.exe`。

无需安装 Python，已包含运行环境、CPU 分割引擎和预训练模型。必须保留 EXE 旁边的 `_internal`、`engine`、`THIRD_PARTY` 文件夹。GitHub 自动生成的 `Source code` 压缩包是源码，不是可直接运行的客户端。

功能包括：

- 导入彩色室内扫描，使用 RandLA-Net 进行语义分割。
- RANSAC 与 DBSCAN 生成候选组，支持人工审阅、改名、合并和拆分。
- 生成贴合点云的彩色网格，导出 Rhino 3DM。
- 图层和对象名称包含中文物体类别、组号及自定义名称。
- 保存与重新打开 `.pcproj` 工程。

本次发布沿用本机已验证的客户端与引擎二进制，更新便携包使用说明，移除本机验证路径。已检查 ZIP 完整性，确认 EXE、引擎和模型权重与原测试包一致。尚未在全新 Windows 设备上验证，具体数据质量与硬件条件会影响结果和耗时。

输出是 Mesh，不保证封闭，不是自动 BIM 或 NURBS 实体。该模型主要面向带 RGB 的室内点云，候选分组需要人工审阅。原始点云数据和示例场景未包含在下载包中。

`SHA256SUMS.txt` 提供客户端压缩包的 SHA-256；在 PowerShell 中可执行 `Get-FileHash .\PointCloudDesk-Windows-x64.zip -Algorithm SHA256` 进行比对。第三方许可和模型来源随包保留在 `THIRD_PARTY`。
