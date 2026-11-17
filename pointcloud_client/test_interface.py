"""In-process Qt widget test: no OS keyboard/mouse input or other app interaction."""
from pathlib import Path
import sys,time,json
from unittest.mock import patch
import numpy as np
from PySide6 import QtWidgets as W,QtCore as C,QtGui as G,QtTest
from app import MainWindow
ROOT=Path(__file__).parent;OUT=ROOT/'qa';OUT.mkdir(exist_ok=True)
app=W.QApplication([]);app.setStyle('Fusion')
window=MainWindow();window.setAttribute(C.Qt.WidgetAttribute.WA_ShowWithoutActivating,True);window.show()
errors=[];messages=[]

def spin_until(predicate,seconds=60):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        app.processEvents();time.sleep(.02)
        if predicate():return
    raise TimeoutError('Qt task did not finish')

def idle():return not (window.worker and window.worker.isRunning()) and window.centralWidget().isEnabled()

with patch.object(W.QMessageBox,'warning',side_effect=lambda *a:errors.append(str(a[-1]))),patch.object(W.QMessageBox,'information',side_effect=lambda *a:messages.append(str(a[-1]))):
    window.open_path(ROOT.parent/'results/area3_batch/Area_3_office_2/segmentation.npz');spin_until(lambda:window.session is not None and idle())
    s=window.session;assert len(s.xyz)==739209
    floor=max((g for g,v in s.info.items() if v['class_id']==1),key=lambda g:sum(s.assignment==g))
    window.pick_group(floor);app.processEvents();assert window.selected()==[floor]
    window.preview_split.setChecked(True);window.split_slider.setValue(45);app.processEvents();assert window.view.cut_item is not None
    original=s.assignment.copy();before=len(s.info);window.split_group();assert len(s.info)==before+1
    window.history(False);np.testing.assert_array_equal(s.assignment,original)
    window.pick_group(floor);window.tabs.setCurrentIndex(1);window.model_kind.setCurrentIndex(window.model_kind.findData('slab'));window.generate_models();spin_until(idle)
    assert len(s.models)==1 and not errors
    # Projective picking uses the same camera matrix as the displayed OpenGL points.
    viewport=window.view.getViewport()
    matrix=np.array((window.view.projectionMatrix(viewport,viewport)*window.view.viewMatrix()).copyDataTo()).reshape(4,4)
    clip=np.c_[window.view.positions,np.ones(len(window.view.positions))]@matrix.T
    ndc=clip[:,:3]/clip[:,3,None];assert ((abs(ndc[:,0])<1)&(abs(ndc[:,1])<1)&(clip[:,3]>0)).sum()>100
    selected=[];window.view.picked.connect(selected.append)
    hit=int(np.flatnonzero((abs(ndc[:,0])<.6)&(abs(ndc[:,1])<.6)&(clip[:,3]>0))[0])
    screen=C.QPoint(int((ndc[hit,0]+1)*window.view.width()/2),int((1-ndc[hit,1])*window.view.height()/2))
    QtTest.QTest.mouseClick(window.view,C.Qt.MouseButton.LeftButton,C.Qt.KeyboardModifier.NoModifier,screen)
    app.processEvents();assert selected and selected[-1] in s.info
    with patch.object(W.QFileDialog,'getSaveFileName',return_value=(str(OUT/'ui_saved.pcproj'),'点云工程')):window.save_dialog();spin_until(idle)
    assert (OUT/'ui_saved.pcproj').exists() and not s.dirty
    with patch.object(W.QFileDialog,'getSaveFileName',return_value=(str(OUT/'ui_export.3dm'),'Rhino 模型')):window.export_dialog();spin_until(idle)
    assert (OUT/'ui_export.3dm').exists() and not errors
    window.isolate.setChecked(False);window.tree.clearSelection();window.render();window.fit_view();app.processEvents()
    # Capture this application's own rendering, without reading the desktop or other windows.
    for _ in range(20):app.processEvents();time.sleep(.05)
    window.grab().save(str(OUT/'client_preview.png'))
    window.view.grabFramebuffer().save(str(OUT/'viewport_preview.png'))
    result={'success':True,'loaded_points':len(s.xyz),'source_groups':before,'widget_selection':True,'split_preview':True,
        'split_and_undo':True,'geometry_generation':True,'project_save':True,'3dm_export':True,'camera_projection':True,'viewport_point_pick':True,'modal_errors':errors}
    (OUT/'interface_test_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
window.session.dirty=False;window.close();app.processEvents()
