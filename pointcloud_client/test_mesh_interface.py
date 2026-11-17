"""Real existing-project -> isolated mesh worker -> editor -> undo/redo test."""
from pathlib import Path
from unittest.mock import patch
import json,time
import numpy as np
from PySide6 import QtCore as C,QtWidgets as W
from app import MainWindow
ROOT=Path(__file__).parent
app=W.QApplication([]);app.setStyle('Fusion');w=MainWindow();w.setAttribute(C.Qt.WidgetAttribute.WA_ShowWithoutActivating,True);w.show();errors=[]
def idle():return not (w.worker and w.worker.isRunning()) and w.centralWidget().isEnabled()
def wait(predicate,timeout=180):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        app.processEvents();time.sleep(.03)
        if predicate():return
    raise TimeoutError('Mesh UI timed out')
with patch.object(W.QMessageBox,'warning',side_effect=lambda *a:errors.append(str(a[-1]))):
    w.open_path(ROOT.parent/'results/area3_batch/Area_3_office_2/segmentation.npz');wait(lambda:w.session is not None and idle())
    original=w.session.assignment.copy();gid=max((g for g,i in w.session.info.items() if i['class_id']==1),key=lambda g:sum(w.session.assignment==g))
    w.pick_group(gid);w.tabs.setCurrentIndex(1);assert w.model_kind.currentData()=='mesh'
    w.generate_models();wait(lambda:errors or (idle() and len(w.session.models)==1));assert not errors,errors
    model=w.session.models[0];assert model['kind']=='mesh' and model['group_id']==gid
    assert not w.show_points.isChecked();assert len(w.view.model_items)==1
    w.show_models.setChecked(False);assert len(w.view.model_items)==0
    w.show_models.setChecked(True);assert len(w.view.model_items)==1
    w.history(False);assert not w.session.models and w.show_points.isChecked()
    w.history(True);assert len(w.session.models)==1
    np.testing.assert_array_equal(w.session.assignment,original)
    result={'success':True,'existing_segmentation_remeshed_without_inference':True,'selected_group':gid,'faces':model['triangles_count'],'undo_redo':True,'display_toggle':True,'assignments_unchanged':True,'saved_model':w.last_scan_result['model'],'errors':errors}
    (ROOT/'qa/mesh_interface_results.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
    w.session.dirty=False;w.close();app.processEvents()
