"""PointCloud Desk: independent offline desktop review and Rhino 3DM export."""
import sys,json,traceback,uuid
from pathlib import Path
import numpy as np
from PySide6 import QtCore as C,QtGui as G,QtWidgets as W
import pyqtgraph.opengl as gl
from core import Session,CLASS_ZH,PALETTE,group_color,fit_candidate,model_vertices
from exporter import export_3dm
from scan_ui import ScanDialog,EngineJob

class Worker(C.QThread):
    result=C.Signal(object);failed=C.Signal(str);progress=C.Signal(int,str)
    def __init__(self,fn,parent=None):super().__init__(parent);self.fn=fn
    def run(self):
        try:self.result.emit(self.fn(self.progress.emit))
        except Exception as exc:self.failed.emit(str(exc));print(traceback.format_exc(),file=sys.stderr)

class CloudView(gl.GLViewWidget):
    picked=C.Signal(int)
    def __init__(self):
        super().__init__();self.setBackgroundColor('#142133');self.positions=np.empty((0,3));self.group_ids=np.empty(0,np.int32)
        self.pressed=None;self.scatter=gl.GLScatterPlotItem(pos=np.zeros((1,3)),size=2,pxMode=True);self.scatter.setGLOptions('opaque');self.addItem(self.scatter)
        self.grid=gl.GLGridItem();self.grid.setColor((90,115,140,65));self.addItem(self.grid);self.model_items=[];self.cut_item=None
    def mousePressEvent(self,event):self.pressed=event.position();super().mousePressEvent(event)
    def mouseReleaseEvent(self,event):
        super().mouseReleaseEvent(event)
        if event.button()!=C.Qt.MouseButton.LeftButton or self.pressed is None or (event.position()-self.pressed).manhattanLength()>4 or not len(self.positions):return
        viewport=self.getViewport()
        matrix=np.array((self.projectionMatrix(viewport,viewport)*self.viewMatrix()).copyDataTo()).reshape(4,4)
        clip=np.c_[self.positions,np.ones(len(self.positions))]@matrix.T
        valid=clip[:,3]>0;ndc=clip[:,:3]/np.maximum(clip[:,3,None],1e-10)
        screen=np.c_[(ndc[:,0]+1)*self.width()/2,(1-ndc[:,1])*self.height()/2]
        distance=((screen-np.array([event.position().x(),event.position().y()]))**2).sum(1)
        distance[~valid|(abs(ndc[:,2])>1)]=np.inf
        i=int(distance.argmin())
        if distance[i]<144:self.picked.emit(int(self.group_ids[i]))
    def set_cloud(self,positions,colors,groups,size):
        self.positions=np.ascontiguousarray(positions,dtype=np.float32);self.group_ids=groups
        self.scatter.setData(pos=self.positions,color=np.ascontiguousarray(colors,dtype=np.float32),size=size,pxMode=True)
    def models(self,models,origin):
        for item in self.model_items:self.removeItem(item)
        self.model_items=[]
        faces=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
        for model in models:
            vertices=model_vertices(model)-origin
            if model['kind']=='mesh':
                colors=np.c_[np.asarray(model['colors'],np.float32)/255.,np.ones(len(vertices),np.float32)]
                data=gl.MeshData(vertexes=vertices,faces=model['faces'],vertexColors=colors)
                mesh=gl.GLMeshItem(meshdata=data,smooth=False,drawEdges=False,shader='shaded',glOptions='opaque')
            else:mesh=gl.GLMeshItem(vertexes=vertices,faces=faces[:2] if model['kind']=='surface' else faces,color=(.1,.84,.69,.46),smooth=False,drawEdges=True,edgeColor=(.3,1,.85,1),glOptions='translucent')
            self.addItem(mesh);self.model_items.append(mesh)
    def split_plane(self,bounds,axis,position,origin,show):
        if self.cut_item is not None:self.removeItem(self.cut_item);self.cut_item=None
        if not show:return
        lo,hi=bounds;other=[i for i in range(3) if i!=axis];a,b=other
        p=np.tile(lo,(5,1));p[:,axis]=position;p[1,a]=hi[a];p[2,a]=hi[a];p[2,b]=hi[b];p[3,b]=hi[b]
        self.cut_item=gl.GLLinePlotItem(pos=p-origin,color=(1,.7,.2,1),width=3,antialias=True);self.addItem(self.cut_item)

class MainWindow(W.QMainWindow):
    def __init__(self,input_path=None):
        super().__init__();self.setWindowTitle('PointCloud Desk v0.3 · 贴合网格');self.resize(1440,930);self.setMinimumSize(1150,760)
        # Explicit light palette prevents Windows dark-mode text on our light panels.
        palette=G.QPalette()
        for role,color in [('Window','#eef2f7'),('WindowText','#25364a'),('Base','#ffffff'),('AlternateBase','#f5f8fb'),('Text','#25364a'),('Button','#ffffff'),('ButtonText','#25364a'),('Highlight','#d8f1ee'),('HighlightedText','#075f58'),('ToolTipBase','#ffffff'),('ToolTipText','#25364a')]:
            palette.setColor(getattr(G.QPalette.ColorRole,role),G.QColor(color))
        W.QApplication.instance().setPalette(palette)
        self.session=None;self.project_path=None;self.worker=None;self.origin=np.zeros(3);self.sample=np.empty(0,int);self.updating=False
        self.build_ui();self.setAcceptDrops(True)
        if input_path:C.QTimer.singleShot(200,lambda:self.open_path(input_path))
    def button(self,text,fn,primary=False):
        b=W.QPushButton(text);b.setMinimumHeight(32);b.clicked.connect(fn)
        if primary:b.setProperty('primary',True)
        return b
    def build_ui(self):
        self.setStyleSheet('''QMainWindow,QWidget#canvas{background:#eef2f7;color:#25364a;font-family:"Microsoft YaHei UI";font-size:12px;} QWidget#header{background:#142133;} QLabel#brand{color:white;font-size:22px;font-weight:700;} QLabel#subbrand{color:#9bb5cc;} QLabel#section{font-size:15px;font-weight:700;color:#17314b;} QFrame#card{background:white;border:1px solid #dce4ed;border-radius:8px;} QPushButton{background:white;border:1px solid #cdd8e3;border-radius:5px;padding:5px 10px;color:#25364a;} QPushButton:hover{background:#e7f3f6;border-color:#66b8b0;} QPushButton[primary="true"]{background:#008c82;color:white;border:1px solid #008c82;font-weight:600;} QPushButton:disabled{color:#a1adba;background:#edf0f3;} QLineEdit,QComboBox,QDoubleSpinBox{border:1px solid #cdd8e3;border-radius:4px;padding:5px;background:white;color:#25364a;min-height:21px;} QTreeWidget{border:0;background:white;alternate-background-color:#f5f8fb;outline:0;} QTreeWidget::item{padding:5px 0;} QTreeWidget::item:selected{background:#d8f1ee;color:#075f58;} QHeaderView::section{background:#f2f6fa;border:0;padding:6px;color:#647990;} QSplitter::handle{background:#eef2f7;width:8px;} QCheckBox{spacing:6px;} QStatusBar{background:#e6edf4;color:#456077;} QTabWidget::pane{border:1px solid #dce4ed;background:white;} QTabBar::tab{padding:9px 16px;background:#e8eef4;} QTabBar::tab:selected{background:white;color:#008c82;}''')
        self.setStyleSheet('QWidget { color:#25364a; font-family:"Microsoft YaHei UI"; } QTreeWidget { color:#25364a; } QTabBar::tab { color:#526a80; } '+self.styleSheet())
        root=W.QWidget();root.setObjectName('canvas');self.setCentralWidget(root);outer=W.QVBoxLayout(root);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        header=W.QWidget();header.setObjectName('header');h=W.QHBoxLayout(header);h.setContentsMargins(20,15,20,15)
        brand=W.QLabel('POINT / FORM');brand.setObjectName('brand');h.addWidget(brand)
        sub=W.QLabel('点云建模工作台  /  扫描到模型');sub.setObjectName('subbrand');h.addWidget(sub);h.addStretch()
        self.scan_button=self.button('导入原始扫描',self.scan_dialog,True);h.addWidget(self.scan_button)
        self.open_button=self.button('打开分割结果',self.open_dialog);self.save_button=self.button('保存工程',self.save_dialog);self.export_button=self.button('导出 Rhino 3DM',self.export_dialog,True)
        h.addWidget(self.open_button);h.addWidget(self.save_button);h.addWidget(self.export_button);outer.addWidget(header)
        tool=W.QHBoxLayout();tool.setContentsMargins(20,10,20,10)
        self.title=W.QLabel('导入原始扫描，或打开已有分割结果生成贴合网格');self.title.setObjectName('section');tool.addWidget(self.title);tool.addStretch()
        tool.addWidget(self.button('撤销',lambda:self.history(False)));tool.addWidget(self.button('重做',lambda:self.history(True)));outer.addLayout(tool)
        splitter=W.QSplitter();outer.addWidget(splitter,1)
        left=W.QFrame();left.setObjectName('card');ll=W.QVBoxLayout(left);ll.setContentsMargins(12,12,12,12)
        title=W.QLabel('01  分组审阅');title.setObjectName('section');ll.addWidget(title)
        self.search=W.QLineEdit();self.search.setPlaceholderText('搜索组名或类别');self.search.textChanged.connect(self.filter_changed);ll.addWidget(self.search)
        self.category=W.QComboBox();self.category.addItem('全部类别',None)
        for i,n in enumerate(CLASS_ZH):self.category.addItem(n,i)
        self.category.addItem('待归组 / 混合',-1);self.category.currentIndexChanged.connect(self.filter_changed);ll.addWidget(self.category)
        self.tree=W.QTreeWidget();self.tree.setHeaderLabels(['分组','点数','分数']);self.tree.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection);self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0,157);self.tree.setColumnWidth(1,76);self.tree.setColumnWidth(2,54)
        self.tree.itemSelectionChanged.connect(self.selection_changed);self.tree.itemChanged.connect(self.visibility_changed);ll.addWidget(self.tree,1)
        hint=W.QLabel('Ctrl / Shift 多选 · 勾选控制可见性\n点击视图中的点也可选中整组');hint.setWordWrap(True);hint.setStyleSheet('color:#71869b;font-size:11px;');ll.addWidget(hint);splitter.addWidget(left)
        middle=W.QWidget();ml=W.QVBoxLayout(middle);ml.setContentsMargins(0,0,0,0);ml.setSpacing(7)
        controls=W.QHBoxLayout();self.color_mode=W.QComboBox();self.color_mode.addItems(['按分组着色','按类别着色','原始 RGB']);self.color_mode.currentIndexChanged.connect(self.render)
        controls.addWidget(self.color_mode);self.cutaway=W.QCheckBox('剖视（仅显示）');self.cutaway.setChecked(True);self.cutaway.toggled.connect(self.render);controls.addWidget(self.cutaway)
        self.isolate=W.QCheckBox('只看选中组');self.isolate.toggled.connect(self.render);controls.addWidget(self.isolate);controls.addStretch();controls.addWidget(self.button('适应视图',self.fit_view));ml.addLayout(controls)
        self.show_points=W.QCheckBox('点云');self.show_points.setChecked(True);self.show_points.toggled.connect(self.render);controls.insertWidget(3,self.show_points)
        self.show_models=W.QCheckBox('模型');self.show_models.setChecked(True);self.show_models.toggled.connect(self.render);controls.insertWidget(4,self.show_models)
        self.view=CloudView();self.view.picked.connect(self.pick_group);ml.addWidget(self.view,1)
        self.view_note=W.QLabel('左键旋转 · 中键平移 · 滚轮缩放   |   数据按米导出');self.view_note.setStyleSheet('color:#627b94;padding:5px;');ml.addWidget(self.view_note);splitter.addWidget(middle)
        right=W.QFrame();right.setObjectName('card');rl=W.QVBoxLayout(right);rl.setContentsMargins(14,14,14,14)
        title=W.QLabel('02  编辑与建模');title.setObjectName('section');rl.addWidget(title)
        self.selection_info=W.QLabel('选择一个或多个候选组');self.selection_info.setWordWrap(True);rl.addWidget(self.selection_info)
        tabs=W.QTabWidget();rl.addWidget(tabs,1)
        edit=W.QWidget();el=W.QVBoxLayout(edit);el.setSpacing(12)
        el.addWidget(W.QLabel('组名'));self.group_name=W.QLineEdit();el.addWidget(self.group_name)
        el.addWidget(W.QLabel('人工确认类别'));self.class_edit=W.QComboBox();self.class_edit.addItem('待归组 / 混合',-1)
        for i,n in enumerate(CLASS_ZH):self.class_edit.addItem(n,i)
        el.addWidget(self.class_edit);el.addWidget(self.button('保存名称与类别',self.rename_group))
        el.addWidget(self.button('合并选中组',self.merge_groups));el.addSpacing(12)
        el.addWidget(W.QLabel('按平面拆分一个组'));self.axis=W.QComboBox();self.axis.addItems(['X 方向','Y 方向','Z 方向']);self.axis.currentIndexChanged.connect(self.reset_split);el.addWidget(self.axis)
        self.split_slider=W.QSlider(C.Qt.Orientation.Horizontal);self.split_slider.setRange(1,99);self.split_slider.setValue(50);self.split_slider.valueChanged.connect(self.slider_changed);el.addWidget(self.split_slider)
        self.split_value=W.QDoubleSpinBox();self.split_value.setRange(-1e8,1e8);self.split_value.setDecimals(3);self.split_value.setSuffix(' m');self.split_value.valueChanged.connect(self.render);el.addWidget(self.split_value)
        self.preview_split=W.QCheckBox('显示切分预览');self.preview_split.toggled.connect(self.render);el.addWidget(self.preview_split)
        el.addWidget(self.button('执行拆分',self.split_group));el.addStretch();tabs.addTab(edit,'分组编辑')
        model=W.QWidget();fm=W.QVBoxLayout(model);fm.setSpacing(12)
        fm.addWidget(W.QLabel('生成方式'));self.model_kind=W.QComboBox();self.model_kind.addItem('贴合点云网格','mesh');self.model_kind.addItem('矩形板体（指定厚度）','slab');self.model_kind.addItem('拟合平面（无厚度）','surface');self.model_kind.addItem('包围盒体块（近似）','box');fm.addWidget(self.model_kind)
        fm.addWidget(W.QLabel('网格采样间距 / 最大连边'))
        mesh_controls=W.QHBoxLayout();self.mesh_spacing=W.QDoubleSpinBox();self.mesh_spacing.setRange(.003,.2);self.mesh_spacing.setDecimals(3);self.mesh_spacing.setValue(.02);self.mesh_spacing.setSuffix(' m')
        self.mesh_edge=W.QDoubleSpinBox();self.mesh_edge.setRange(.006,1);self.mesh_edge.setDecimals(3);self.mesh_edge.setValue(.08);self.mesh_edge.setSuffix(' m');mesh_controls.addWidget(self.mesh_spacing);mesh_controls.addWidget(self.mesh_edge);fm.addLayout(mesh_controls)
        fm.addWidget(W.QLabel('板体厚度'));self.thickness=W.QDoubleSpinBox();self.thickness.setRange(.001,20);self.thickness.setDecimals(3);self.thickness.setValue(.2);self.thickness.setSuffix(' m');fm.addWidget(self.thickness)
        fm.addWidget(W.QLabel('95%点允许的平面偏差'));self.tolerance=W.QDoubleSpinBox();self.tolerance.setRange(.001,2);self.tolerance.setDecimals(3);self.tolerance.setValue(.04);self.tolerance.setSuffix(' m');fm.addWidget(self.tolerance)
        caution=W.QLabel('网格沿实测点生成，保留原始颜色。\n间距越小越精细；连边过大可能跨过缺口。\n允许开口，不自动封洞或添加厚度。\n板体/包围盒仍是旧版简化几何。');caution.setWordWrap(True);caution.setStyleSheet('color:#8a662b;background:#fff6e5;padding:9px;border-radius:5px;');fm.addWidget(caution)
        fm.addWidget(self.button('为选中组生成几何',self.generate_models,True));fm.addWidget(self.button('清空生成几何',self.clear_models))
        fm.addWidget(self.button('全部分组重建网格',lambda:self.generate_mesh_job(list(self.session.info) if self.session else [])))
        self.model_info=W.QLabel('尚未生成几何');self.model_info.setWordWrap(True);fm.addWidget(self.model_info)
        self.include_clouds=W.QCheckBox('导出完整点云参考');self.include_clouds.setChecked(True);fm.addWidget(self.include_clouds);fm.addStretch();tabs.addTab(model,'几何生成')
        self.model_kind.currentIndexChanged.connect(self.geometry_mode_changed);self.geometry_mode_changed()
        self.tabs=tabs;splitter.addWidget(right);splitter.setSizes([310,760,320]);left.setMinimumWidth(280);right.setMinimumWidth(285)
        self.progress=W.QProgressBar();self.progress.setMaximumWidth(190);self.progress.setVisible(False);self.statusBar().addPermanentWidget(self.progress);self.statusBar().showMessage('就绪 · 在本机处理，无需连接 Rhino')
        self.cancel_button=self.button('取消处理',self.cancel_scan);self.cancel_button.hide();self.statusBar().addPermanentWidget(self.cancel_button)
        self.results_button=self.button('打开结果文件夹',self.open_results);self.results_button.hide();self.statusBar().addPermanentWidget(self.results_button)
        for key,fn in [('Ctrl+O',self.open_dialog),('Ctrl+S',self.save_dialog),('Ctrl+Z',lambda:self.history(False)),('Ctrl+Shift+Z',lambda:self.history(True))]:
            shortcut=G.QShortcut(G.QKeySequence(key),self);shortcut.activated.connect(fn)
    def selected(self):return [int(i.data(0,C.Qt.ItemDataRole.UserRole)) for i in self.tree.selectedItems() if i.data(0,C.Qt.ItemDataRole.UserRole) is not None]
    def error(self,text):W.QMessageBox.warning(self,'请检查',str(text))
    def run_task(self,fn,done,message):
        if self.worker and self.worker.isRunning():return
        self.centralWidget().setEnabled(False);self.progress.setVisible(True);self.progress.setRange(0,0);self.statusBar().showMessage(message)
        self.worker=Worker(fn,self);self.worker.result.connect(done);self.worker.failed.connect(self.error)
        self.worker.progress.connect(lambda n,t:(self.progress.setRange(0,100),self.progress.setValue(n),self.statusBar().showMessage(t)))
        self.worker.finished.connect(self.task_finished);self.worker.start()
    def task_finished(self):self.centralWidget().setEnabled(True);self.progress.setVisible(False);self.cancel_button.hide()
    def geometry_mode_changed(self):
        mesh=self.model_kind.currentData()=='mesh'
        for widget in [self.mesh_spacing,self.mesh_edge]:widget.setEnabled(mesh)
        self.thickness.setEnabled(not mesh);self.tolerance.setEnabled(not mesh)
    def scan_dialog(self,path=None):
        if self.worker and self.worker.isRunning():return
        if not self.can_replace():return
        dialog=ScanDialog(self,path if isinstance(path,(str,Path)) else None)
        if dialog.exec()==W.QDialog.DialogCode.Accepted:self.start_scan(dialog.settings())
    def start_scan(self,config):
        if self.worker and self.worker.isRunning():return
        self.centralWidget().setEnabled(False);self.progress.setVisible(True);self.progress.setRange(0,0);self.cancel_button.show()
        self.statusBar().showMessage('正在启动内置分割引擎…');self.worker=EngineJob(config,self)
        self.worker.progress.connect(lambda n,t:(self.progress.setRange(0,100),self.progress.setValue(n),self.statusBar().showMessage(t)))
        self.worker.failed.connect(self.error);self.worker.finished.connect(self.task_finished);self.worker.result.connect(self.mesh_completed if config.get('task')=='remesh' else self.scan_completed)
        try:self.worker.start()
        except Exception as e:self.task_finished();self.error(str(e))
    def cancel_scan(self):
        if isinstance(self.worker,EngineJob):self.worker.cancel()
    def scan_completed(self,result):
        self.last_scan_result=result;self.results_button.show()
        # Wait for EngineJob.finished before starting the normal project loader.
        C.QTimer.singleShot(0,lambda:self.open_path(result['project'],confirmed=True))
    def generate_mesh_job(self,ids):
        if not self.session or not ids:return self.error('请先打开点云并选择需要重建的分组。')
        if self.worker and self.worker.isRunning():return
        if self.mesh_edge.value()<2*self.mesh_spacing.value():return self.error('最大连边长度需至少为采样间距的 2 倍。')
        parent=Path.home()/'Documents/PointCloudDeskResults';parent.mkdir(parents=True,exist_ok=True)
        name='mesh_'+uuid.uuid4().hex[:12];source=parent/(name+'_input.pcproj')
        config=dict(task='remesh',input=str(source),output=str(parent/name),group_ids=ids,mesh_spacing=self.mesh_spacing.value(),mesh_max_edge=self.mesh_edge.value())
        session=self.session;dirty=session.dirty
        def save_input(_):
            try:session.save(source)
            finally:session.dirty=dirty
        self.run_task(save_input,lambda _:C.QTimer.singleShot(0,lambda:self.start_scan(config)),'保存当前分组，准备重建网格…')
    def mesh_completed(self,result):
        def apply_models(updated):
            self.session.checkpoint('reconstruct measured-surface meshes');self.session.models=updated.models
            self.session.metadata['mesh_reconstruction']=result['report']['geometry']
            self.last_scan_result=result;self.results_button.show();self.show_points.setChecked(False);self.show_models.setChecked(True);self.render()
            report=result['report']['geometry'];self.statusBar().showMessage(f"生成 {report['meshes']} 个网格、{report['triangles']:,} 个三角面；可撤销。3DM 已保存到结果文件夹。")
        C.QTimer.singleShot(0,lambda:self.run_task(lambda _:Session.load(result['project']),apply_models,'加载重建网格…'))
    def open_results(self):
        if hasattr(self,'last_scan_result'):G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(Path(self.last_scan_result['project']).parent)))
    def can_replace(self):
        return not (self.session and self.session.dirty) or W.QMessageBox.question(self,'未保存的修改','当前工程尚未保存，放弃修改并继续？',W.QMessageBox.StandardButton.Discard|W.QMessageBox.StandardButton.Cancel)==W.QMessageBox.StandardButton.Discard
    def open_dialog(self):
        path,_=W.QFileDialog.getOpenFileName(self,'打开分割结果或工程','','分割结果 / 工程 (*.npz *.pcproj)')
        if path:self.open_path(path)
    def open_path(self,path,confirmed=False):
        if Path(path).suffix.lower() not in ['.npz','.pcproj']:return self.scan_dialog(path)
        if Path(path).suffix.lower()=='.npz':
            try:
                with np.load(path,allow_pickle=False) as raw:
                    if not {'semantic','confidence','group_id'}.issubset(raw.files):return self.scan_dialog(path)
            except (ValueError,OSError):pass  # The normal loader reports invalid files.
        if not confirmed and not self.can_replace():return
        def loaded(session):
            self.session=session;self.project_path=Path(path) if str(path).endswith('.pcproj') else None
            self.show_points.blockSignals(True);self.show_points.setChecked(not any(m['kind']=='mesh' for m in session.models));self.show_points.blockSignals(False)
            self.origin=(session.xyz.min(0)+session.xyz.max(0))/2
            self.sample=np.random.default_rng(42).choice(len(session.xyz),min(140000,len(session.xyz)),replace=False)
            self.title.setText(f"{session.metadata.get('room',Path(path).stem)}  ·  {len(session.xyz):,} 点")
            self.rebuild_tree();self.fit_view();self.render();self.statusBar().showMessage(f'已加载 {len(session.info)} 个候选组 · 所有编辑与导出使用完整点云')
            if confirmed and hasattr(self,'last_scan_result'):
                summary=self.last_scan_result['report']['export'];self.statusBar().showMessage(f"扫描处理完成：{summary.get('meshes',0)} 个网格、{summary['cloud_objects']} 个点云组；3DM 已保存，请复核后使用")
        self.run_task(lambda _:Session.load(path),loaded,'正在加载点云…')
    def rebuild_tree(self,select=None):
        if not self.session:return
        self.updating=True;self.tree.clear();parents={};counts=dict(zip(*np.unique(self.session.assignment,return_counts=True)))
        for gid,info in sorted(self.session.info.items(),key=lambda kv:(kv[1]['class_id'],kv[0])):
            cid=info['class_id']
            if cid not in parents:
                parent=W.QTreeWidgetItem(self.tree,[CLASS_ZH[cid] if cid>=0 else '待归组 / 混合']);parent.setFlags(parent.flags()&~C.Qt.ItemFlag.ItemIsSelectable);parent.setExpanded(True);parents[cid]=parent
            conf=float(self.session.confidence[self.session.assignment==gid].mean())
            item=W.QTreeWidgetItem(parents[cid],[info['name'],f'{counts[gid]:,}',f'{conf:.0%}']);item.setData(0,C.Qt.ItemDataRole.UserRole,gid)
            item.setFlags(item.flags()|C.Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(0,C.Qt.CheckState.Checked if info['visible'] else C.Qt.CheckState.Unchecked)
            item.setForeground(0,G.QBrush(G.QColor(*map(int,group_color(gid)*200))))
            if select and gid in select:item.setSelected(True)
        self.updating=False;self.filter_changed();self.selection_changed()
    def filter_changed(self,*_):
        if not self.session:return
        text=self.search.text().strip().lower();cid=self.category.currentData()
        for i in range(self.tree.topLevelItemCount()):
            parent=self.tree.topLevelItem(i);any_visible=False
            for j in range(parent.childCount()):
                item=parent.child(j);gid=int(item.data(0,C.Qt.ItemDataRole.UserRole));info=self.session.info[gid]
                match=(cid is None or cid==info['class_id']) and (not text or text in (info['name']+' '+parent.text(0)).lower());item.setHidden(not match);any_visible|=match
            parent.setHidden(not any_visible)
        self.render()
    def visible_ids(self):
        gids=[]
        for i in range(self.tree.topLevelItemCount()):
            parent=self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                item=parent.child(j)
                if not parent.isHidden() and not item.isHidden() and item.checkState(0)==C.Qt.CheckState.Checked:gids.append(int(item.data(0,C.Qt.ItemDataRole.UserRole)))
        return gids
    def visibility_changed(self,item,column):
        if self.updating or not self.session:return
        gid=item.data(0,C.Qt.ItemDataRole.UserRole)
        if gid is not None:self.session.info[int(gid)]['visible']=item.checkState(0)==C.Qt.CheckState.Checked
        self.render()
    def selection_changed(self):
        if self.updating or not self.session:return
        selected=self.selected();n=len(self.session.indices(selected))
        self.selection_info.setText(f'已选 {len(selected)} 组 · {n:,} 点')
        if len(selected)==1:
            info=self.session.info[selected[0]];self.group_name.setText(info['name']);self.class_edit.setCurrentIndex(self.class_edit.findData(info['class_id']))
        self.reset_split();self.render()
    def pick_group(self,gid):
        if not self.session:return
        self.tree.blockSignals(True);self.tree.clearSelection()
        for i in range(self.tree.topLevelItemCount()):
            parent=self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                item=parent.child(j)
                if item.data(0,C.Qt.ItemDataRole.UserRole)==gid:item.setSelected(True);self.tree.scrollToItem(item)
        self.tree.blockSignals(False);self.selection_changed()
    def reset_split(self,*_):
        if not self.session:return
        selected=self.selected()
        if len(selected)!=1:return
        p=self.session.xyz[self.session.assignment==selected[0]];axis=self.axis.currentIndex();self.split_range=(float(p[:,axis].min()),float(p[:,axis].max()))
        self.split_value.blockSignals(True);self.split_value.setValue(sum(self.split_range)/2);self.split_value.blockSignals(False)
        self.split_slider.blockSignals(True);self.split_slider.setValue(50);self.split_slider.blockSignals(False);self.render()
    def slider_changed(self,v):
        if hasattr(self,'split_range'):lo,hi=self.split_range;self.split_value.setValue(lo+(hi-lo)*v/100)
    def render(self,*_):
        if not self.session:return
        s=self.session;ids=self.visible_ids();selected=self.selected();idx=self.sample
        mask=np.isin(s.assignment[idx],selected if self.isolate.isChecked() else ids)
        if not self.show_points.isChecked():mask[:]=False
        if self.cutaway.isChecked() and not self.isolate.isChecked():
            lo=s.xyz.min(0);hi=s.xyz.max(0);p=s.xyz[idx]
            ceiling=[g for g,v in s.info.items() if v['class_id']==0]
            mask&=~np.isin(s.assignment[idx],ceiling);mask&=(p[:,1]>lo[1]+.06)&(p[:,0]<hi[0]-.06)
        idx=idx[mask];p=s.xyz[idx];gids=s.assignment[idx];mode=self.color_mode.currentIndex()
        if mode==2:colors=s.rgb[idx]/255.
        elif mode==1:
            colors=np.array([PALETTE[s.info[int(g)]['class_id']]/255. if s.info[int(g)]['class_id']>=0 else [.55,.59,.63] for g in gids])
        else:
            lookup={g:group_color(g) for g in set(map(int,gids))};colors=np.array([lookup[int(g)] for g in gids])
        colors=colors.reshape(-1,3)
        if selected:
            chosen=np.isin(gids,selected);colors[~chosen]=colors[~chosen]*.25+np.array([.08,.12,.18])*.75;colors[chosen]=[1.,.75,.3]
        show=self.preview_split.isChecked() and len(selected)==1
        if show:
            chosen=gids==selected[0];right=p[:,self.axis.currentIndex()]>self.split_value.value();colors[chosen&right]=[.1,.85,.77];colors[chosen&~right]=[1.,.67,.23]
        self.view.set_cloud(p-self.origin,np.c_[colors,np.ones(len(colors))],gids,2.2)
        model_ids=set(selected if self.isolate.isChecked() else ids)
        if self.cutaway.isChecked() and not self.isolate.isChecked():
            model_ids-={g for g,v in s.info.items() if v['class_id']==0}
        self.view.models([m for m in s.models if m['group_id'] in model_ids] if self.show_models.isChecked() else [],self.origin)
        if selected:
            region=s.xyz[s.indices(selected)];bounds=(region.min(0),region.max(0))
        else:bounds=(s.xyz.min(0),s.xyz.max(0))
        self.view.split_plane(bounds,self.axis.currentIndex(),self.split_value.value(),self.origin,show)
        self.view_note.setText(f'显示 {len(p):,} / {len(s.xyz):,} 点（显示抽样） · 编辑与导出使用完整点云 · 单位 m')
        details=[m for m in s.models if m['group_id'] in selected] if selected else s.models[-3:]
        self.model_info.setText(f'已生成 {len(s.models)} 个可编辑几何\n'+('\n'.join((f"{m['name']} · {m['triangles_count']:,} 面\n点到面95% {m['cloud_to_surface_p95_m']:.3f} m" if m['kind']=='mesh' else f"{m['name']} · 偏差95% {m['p95_m']:.3f}m") for m in details[:3]) if s.models else '选择分组后生成'))
    def fit_view(self):
        if not self.session:return
        selected=self.selected();p=self.session.xyz[self.session.indices(selected)] if self.isolate.isChecked() and selected else self.session.xyz
        lo=p.min(0);hi=p.max(0);center=(lo+hi)/2-self.origin;distance=max(np.linalg.norm(hi-lo)*1.45,1.)
        self.view.setCameraPosition(pos=G.QVector3D(*center),distance=distance,elevation=28,azimuth=-60)
        self.view.grid.setSize(x=max(4,float(hi[0]-lo[0])*1.5),y=max(4,float(hi[1]-lo[1])*1.5));self.view.grid.resetTransform();self.view.grid.translate(0,0,float(lo[2]-self.origin[2]-.02))
    def rename_group(self):
        if not self.session or len(self.selected())!=1:return self.error('请选择一个分组。')
        try:gid=self.selected()[0];self.session.rename(gid,self.group_name.text(),self.class_edit.currentData());self.rebuild_tree([gid])
        except Exception as e:self.error(e)
    def merge_groups(self):
        if not self.session:return
        try:
            new=self.session.merge(self.selected(),self.class_edit.currentData());self.rebuild_tree([new]);self.statusBar().showMessage('已合并；原始模型预测保留，可撤销')
        except Exception as e:self.error(e)
    def split_group(self):
        if not self.session or len(self.selected())!=1:return self.error('请选择一个分组。')
        try:ids=self.session.split(self.selected()[0],self.axis.currentIndex(),self.split_value.value());self.preview_split.setChecked(False);self.rebuild_tree(ids);self.statusBar().showMessage('已拆分为两个点云组，可撤销')
        except Exception as e:self.error(e)
    def history(self,redo):
        if self.session and self.session.undo(redo):
            if not self.session.models:self.show_points.setChecked(True)
            self.rebuild_tree();self.statusBar().showMessage('已重做' if redo else '已撤销')
    def generate_models(self):
        if not self.session or not self.selected():return self.error('请先选择要建模的分组。')
        if self.model_kind.currentData()=='mesh':return self.generate_mesh_job(self.selected())
        ids=self.selected();kind=self.model_kind.currentData();thickness=self.thickness.value();tolerance=self.tolerance.value();s=self.session
        def work(progress):
            models=[];errors=[]
            for i,gid in enumerate(ids):
                try:models.append(fit_candidate(s.xyz[s.assignment==gid],gid,s.info[gid]['name'],kind,thickness,tolerance))
                except ValueError as e:errors.append(s.info[gid]['name']+'：'+str(e))
                progress(int((i+1)/len(ids)*100),'正在拟合选中组')
            return models,errors
        def done(result):
            models,errors=result
            if models:s.checkpoint('generate '+kind);s.invalidate({m['group_id'] for m in models});s.models.extend(models);self.render()
            self.statusBar().showMessage(f'生成 {len(models)} 个几何；{len(errors)} 组未通过拟合检查')
            if errors:self.error('\n'.join(errors[:6]))
        self.run_task(work,done,'正在生成几何…')
    def clear_models(self):
        if self.session:self.session.clear_models();self.show_points.setChecked(True);self.render()
    def save_dialog(self):
        if not self.session:return self.error('请先打开分割结果。')
        path,_=W.QFileDialog.getSaveFileName(self,'保存可继续编辑的工程',str(self.project_path or Path.home()/'Desktop'/'点云编辑.pcproj'),'点云工程 (*.pcproj)')
        if not path:return
        if not path.lower().endswith('.pcproj'):path+='.pcproj'
        self.project_path=Path(path)
        self.run_task(lambda _:self.session.save(path),lambda _:self.statusBar().showMessage('工程已保存：'+path),'正在保存工程…')
    def export_dialog(self):
        if not self.session:return self.error('请先打开分割结果。')
        if not self.session.models and not self.include_clouds.isChecked():return self.error('没有可导出的几何；请生成模型或勾选点云参考。')
        path,_=W.QFileDialog.getSaveFileName(self,'导出 Rhino 3DM',str(Path.home()/'Desktop'/'点云建模.3dm'),'Rhino 模型 (*.3dm)')
        if not path:return
        if not path.lower().endswith('.3dm'):path+='.3dm'
        include=self.include_clouds.isChecked()
        def done(result):
            self.statusBar().showMessage(f"已导出并回读验证：{result['models']} 个几何、{result['cloud_objects']} 个点云组")
            W.QMessageBox.information(self,'导出成功',f"已保存 {result.get('meshes',0)} 个网格、{result['closed_solids']} 个闭合板体和 {result['cloud_objects']} 个点云组。\n网格允许开口，不代表闭合实体。\n\n{path}\n\n点云参考图层默认隐藏，可在 Rhino 中开启。")
        self.run_task(lambda progress:export_3dm(self.session,path,include,progress),done,'正在导出 Rhino 文件…')
    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].toLocalFile().lower().endswith(('.npz','.pcproj','.pcd','.ply','.xyz','.xyzrgb','.pts','.txt')):event.acceptProposedAction()
    def dropEvent(self,event):self.open_path(event.mimeData().urls()[0].toLocalFile())
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():event.ignore();self.statusBar().showMessage('正在处理，请完成后再关闭。');return
        if not self.can_replace():event.ignore();return
        event.accept()

def main():
    if len(sys.argv)>2 and sys.argv[1]=='--scan-smoke':
        config=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'));out=Path(config['output'])
        app=W.QApplication(sys.argv);app.setStyle('Fusion');window=MainWindow();window.show()
        def fail(message):
            out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.ui_error.txt').write_text(str(message),encoding='utf8');app.exit(1)
        window.error=fail
        def check_done():
            if hasattr(window,'last_scan_result') and window.session is not None and not window.worker.isRunning() and window.centralWidget().isEnabled():
                summary=dict(window.last_scan_result['report']['export']);summary.update(success=True,packaged=bool(getattr(sys,'frozen',False)),gui_loaded_points=len(window.session.xyz),gui_models=len(window.session.models))
                (out/'packaged_scan_ui.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
                window.grab().save(str(out/'client.png'));app.quit()
        timer=C.QTimer(window);timer.timeout.connect(check_done);timer.start(100)
        C.QTimer.singleShot(0,lambda:window.start_scan(config));C.QTimer.singleShot(300000,lambda:fail('Scan test timed out'))
        return app.exec()
    if len(sys.argv)>3 and sys.argv[1]=='--smoke-test':
        out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
        try:
            s=Session.load(sys.argv[2])
            floor=max((g for g,v in s.info.items() if v['class_id']==1),key=lambda g:np.sum(s.assignment==g))
            s.models=[fit_candidate(s.xyz[s.assignment==floor],floor,s.info[floor]['name'])]
            s.save(out/'packaged_saved.pcproj')
            restored=Session.load(out/'packaged_saved.pcproj')
            assert np.array_equal(s.assignment,restored.assignment)
            result=export_3dm(restored,out/'packaged_export.3dm',True)
            result.update(success=True,project_roundtrip=True,packaged=bool(getattr(sys,'frozen',False)))
            (out/'packaged_test_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            return 0
        except Exception:
            (out/'packaged_test_error.txt').write_text(traceback.format_exc(),encoding='utf8')
            return 1
    app=W.QApplication(sys.argv);app.setApplicationName('PointCloud Desk');app.setStyle('Fusion')
    window=MainWindow(sys.argv[1] if len(sys.argv)>1 and Path(sys.argv[1]).exists() else None);window.show();return app.exec()

if __name__=='__main__':sys.exit(main())
