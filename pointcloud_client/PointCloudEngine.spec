# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all,collect_dynamic_libs
root=Path(SPECPATH)
data,binary,hidden=collect_all('rhino3dm')
binary += [entry for entry in collect_dynamic_libs('open3d') if 'torch_ops' not in entry[0]]
data += [(str(root.parent/'weights/randlanet_s3dis.pth'),'weights'),
         (str(root.parent/'vendor/open3d_ml/LICENSE'),'licenses/open3d_ml')]
a=Analysis(['engine_main.py'],pathex=[str(root.parent)],binaries=binary,datas=data,
           hiddenimports=hidden+['uuid'],
           excludes=['open3d.ml.torch','open3d.ml.tf','open3d._ml3d','mitsuba','matplotlib','pandas','cv2','PySide6','PySide2','PyQt5','PyQt6',
                     'tensorflow','jax','torchvision','torchaudio','IPython','notebook','pytest','tensorboard'],
           noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='PointCloudEngine',console=True,upx=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='PointCloudEngine')
