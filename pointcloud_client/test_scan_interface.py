"""Run the actual model from the GUI controller on an unlabeled real scan."""
from pathlib import Path
import json,time,sys
from unittest.mock import patch
import numpy as np
from plyfile import PlyData,PlyElement
from PySide6 import QtWidgets as W,QtCore as C
from app import MainWindow
from scan_ui import ScanDialog,EngineJob
import rhino3dm as r

ROOT=Path(__file__).parent;OUT=ROOT/'qa';OUT.mkdir(exist_ok=True)
source=ROOT.parent/'data/full_processed/Area_3_office_2.npz'
with np.load(source) as raw:
    xyz=raw['xyz'];rgb=raw['rgb'];truth=raw['semantic']
vertex=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
for i,k in enumerate(['x','y','z']):vertex[k]=xyz[:,i]
for i,k in enumerate(['red','green','blue']):vertex[k]=rgb[:,i]
scan=OUT/'raw_office2.ply';PlyData([PlyElement.describe(vertex,'vertex')],text=False).write(scan)
app=W.QApplication([]);app.setStyle('Fusion');window=MainWindow();window.setAttribute(C.Qt.WidgetAttribute.WA_ShowWithoutActivating,True);window.show()
errors=[]
def spin(predicate,timeout=300):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents();time.sleep(.03)
        if predicate():return
    raise TimeoutError('Scan UI task timed out')
def idle():return not (window.worker and window.worker.isRunning()) and window.centralWidget().isEnabled()
with patch.object(W.QMessageBox,'warning',side_effect=lambda *a:errors.append(str(a[-1]))):
    dialog=ScanDialog(window,scan);dialog.output.setText(str(OUT/'scan_jobs'));dialog.reconstruction.setCurrentIndex(dialog.reconstruction.findData('slab'));config=dialog.settings();dialog.close()
    assert config['auto_model'] and not config['neutral'] and not config['boxes']
    window.start_scan(config);spin(lambda:idle() and (window.session is not None or errors))
    assert not errors,errors
    s=window.session;assert len(s.xyz)==len(xyz) and len(s.models)>0
    target=Path(window.last_scan_result['model']);doc=r.File3dm.Read(str(target));assert doc
    clouds=[o.Geometry for o in doc.Objects if isinstance(o.Geometry,r.PointCloud)]
    solids=[o.Geometry for o in doc.Objects if isinstance(o.Geometry,r.Brep)]
    assert sum(len(c) for c in clouds)==len(xyz)
    assert all(b.IsValid and b.IsSolid for b in solids)
    assert all(o.Geometry.IsValid for o in doc.Objects)
    identity=id(s);cancel_config=dict(config,output=config['output']+'_cancel')
    window.start_scan(cancel_config);C.QTimer.singleShot(400,window.cancel_scan);spin(idle)
    assert id(window.session)==identity and not errors
    assert not (Path(cancel_config['output'])/'complete.json').exists()
    result={'success':True,'raw_input':str(scan),'input_has_only_xyz_rgb':True,'points':len(xyz),'groups':len(s.info),'generated_solids':len(solids),'all_geometry_valid':True,'full_points_preserved':True,'cancel_preserved_editor':True,'project_loaded_in_gui':True,'model_path':str(target),'project_path':str(window.project_path),'observed_accuracy_on_this_room':float(np.mean(s.prediction==truth)),'errors':errors}
    (OUT/'scan_interface_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    window.tabs.setCurrentIndex(1);window.fit_view()
    for _ in range(15):app.processEvents();time.sleep(.05)
    window.grab().save(str(OUT/'integrated_client_preview.png'))
    print(json.dumps(result,indent=2),flush=True)
window.session.dirty=False;window.close();app.processEvents()
