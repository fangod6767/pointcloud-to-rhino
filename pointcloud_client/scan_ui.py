"""Qt scan settings and a cancellable isolated inference process."""
from pathlib import Path
from datetime import datetime
import sys,json,uuid
from PySide6 import QtCore as C,QtWidgets as W

class ScanDialog(W.QDialog):
    def __init__(self,parent=None,path=None):
        super().__init__(parent);self.setWindowTitle('导入扫描 → 分割 → Rhino 模型');self.resize(610,540)
        layout=W.QVBoxLayout(self)
        intro=W.QLabel('适用于室内建筑扫描。先识别类别、分组，再重建贴合点云的网格。\n网格保留实测颜色，缺失处可能开口，不保证闭合实体。')
        intro.setWordWrap(True);layout.addWidget(intro)
        form=W.QFormLayout();layout.addLayout(form)
        self.input=W.QLineEdit(str(path or ''));row=W.QHBoxLayout();row.addWidget(self.input)
        choose=W.QPushButton('选择文件');choose.clicked.connect(self.choose_input);row.addWidget(choose);form.addRow('原始点云',row)
        self.output=W.QLineEdit(str(Path.home()/'Documents/PointCloudDeskResults'));row=W.QHBoxLayout();row.addWidget(self.output)
        choose=W.QPushButton('选择目录');choose.clicked.connect(self.choose_output);row.addWidget(choose);form.addRow('结果保存位置',row)
        self.units=W.QComboBox();self.units.addItem('米 m',1.);self.units.addItem('毫米 mm',.001);self.units.addItem('厘米 cm',.01);form.addRow('原始坐标单位',self.units)
        self.axis=W.QComboBox()
        for a in ['z','y','x']:self.axis.addItem(a.upper()+' 轴向上',a)
        form.addRow('竖直方向',self.axis)
        self.neutral=W.QCheckBox('无颜色实验模式（填充灰色，识别效果尚未验证）');form.addRow(self.neutral)
        self.auto=W.QCheckBox('自动生成几何，并导出 3DM');self.auto.setChecked(True);form.addRow(self.auto)
        self.reconstruction=W.QComboBox();self.reconstruction.addItem('贴合点云网格（推荐）','mesh');self.reconstruction.addItem('简化板体 / 包围盒（旧版）','slab');form.addRow('重建方式',self.reconstruction)
        self.mesh_spacing=W.QDoubleSpinBox();self.mesh_spacing.setRange(.003,.2);self.mesh_spacing.setDecimals(3);self.mesh_spacing.setValue(.02);self.mesh_spacing.setSuffix(' m');form.addRow('网格采样间距',self.mesh_spacing)
        self.mesh_edge=W.QDoubleSpinBox();self.mesh_edge.setRange(.006,1);self.mesh_edge.setDecimals(3);self.mesh_edge.setValue(.08);self.mesh_edge.setSuffix(' m');form.addRow('最大连边长度',self.mesh_edge)
        self.boxes=W.QCheckBox('其他组使用近似包围盒占位（可能与物体差别很大）');form.addRow(self.boxes)
        self.thickness=W.QDoubleSpinBox();self.thickness.setRange(.001,20);self.thickness.setDecimals(3);self.thickness.setValue(.2);self.thickness.setSuffix(' m');form.addRow('板体厚度',self.thickness)
        self.tolerance=W.QDoubleSpinBox();self.tolerance.setRange(.001,2);self.tolerance.setDecimals(3);self.tolerance.setValue(.04);self.tolerance.setSuffix(' m');form.addRow('95% 点允许平面偏差',self.tolerance)
        self.reconstruction.currentIndexChanged.connect(self.update_mode);self.update_mode()
        note=W.QLabel('间距越小越细致，也越慢；连边越大越可能跨过缺口。\n默认 0.02 m / 0.08 m 适合先试室内扫描。\n完整点云、网格、工程与报告均保存；不修改输入文件。')
        note.setWordWrap(True);layout.addWidget(note);layout.addStretch()
        buttons=W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok|W.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(W.QDialogButtonBox.StandardButton.Ok).setText('开始处理');buttons.button(W.QDialogButtonBox.StandardButton.Cancel).setText('取消');buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def update_mode(self):
        mesh=self.reconstruction.currentData()=='mesh'
        for w in [self.mesh_spacing,self.mesh_edge]:w.setEnabled(mesh)
        for w in [self.boxes,self.thickness,self.tolerance]:w.setEnabled(not mesh)
    def choose_input(self):
        path,_=W.QFileDialog.getOpenFileName(self,'选择原始扫描','','点云 (*.pcd *.ply *.xyz *.xyzrgb *.pts *.txt *.npz)')
        if path:self.input.setText(path)
    def choose_output(self):
        path=W.QFileDialog.getExistingDirectory(self,'选择结果目录',self.output.text())
        if path:self.output.setText(path)
    def accept(self):
        if not Path(self.input.text()).is_file():W.QMessageBox.warning(self,'检查输入','请选择存在的点云文件');return
        if not self.output.text().strip():W.QMessageBox.warning(self,'检查输入','请选择结果保存目录');return
        if self.reconstruction.currentData()=='mesh' and self.mesh_edge.value()<2*self.mesh_spacing.value():W.QMessageBox.warning(self,'检查参数','最大连边长度需至少为采样间距的 2 倍');return
        super().accept()
    def settings(self):
        base=Path(self.output.text());name=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        return dict(input=str(Path(self.input.text()).resolve()),output=str(base.resolve()/name),scale=self.units.currentData(),up_axis=self.axis.currentData(),neutral=self.neutral.isChecked(),auto_model=self.auto.isChecked(),boxes=self.boxes.isChecked(),thickness=self.thickness.value(),tolerance=self.tolerance.value(),reconstruction=self.reconstruction.currentData(),mesh_spacing=self.mesh_spacing.value(),mesh_max_edge=self.mesh_edge.value())

class EngineJob(C.QObject):
    result=C.Signal(object);failed=C.Signal(str);progress=C.Signal(int,str);finished=C.Signal()
    def __init__(self,config,parent=None):
        super().__init__(parent);self.config=config;self.process=C.QProcess(self);self.buffer='';self.final=None;self.error=None;self.cancelled=False;self.ended=False;self.log=[]
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.on_finished);self.process.errorOccurred.connect(self.process_error)
    def isRunning(self):return self.process.state()!=C.QProcess.ProcessState.NotRunning
    def start(self):
        out=Path(self.config['output']);out.parent.mkdir(parents=True,exist_ok=True)
        self.config_path=out.with_name('.'+out.name+'.job.json');self.config_path.write_text(json.dumps(self.config),encoding='utf8')
        root=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
        if getattr(sys,'frozen',False):
            exe=root/'engine/PointCloudEngine.exe'
            if not exe.exists():self.failed.emit('缺少内置分割引擎，请完整解压包含 engine 目录的完整版客户端。');self.finished.emit();return
            self.process.setProgram(str(exe));self.process.setArguments(['--config',str(self.config_path)])
        else:
            self.process.setProgram(sys.executable);self.process.setArguments(['-u',str(root/'engine_main.py'),'--config',str(self.config_path)])
        environment=C.QProcessEnvironment.systemEnvironment();environment.insert('PYTHONUTF8','1');environment.insert('OPENBLAS_NUM_THREADS','8');self.process.setProcessEnvironment(environment)
        self.process.start()
    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1);self.log.append(line)
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('event')=='progress':self.progress.emit(event['percent'],event['message'])
            elif event.get('event')=='complete':self.final=event
            elif event.get('event')=='error':self.error=event['message']
    def read_error(self):self.log.append(bytes(self.process.readAllStandardError()).decode('utf8',errors='replace'))
    def process_error(self,error):
        if error==C.QProcess.ProcessError.FailedToStart:self.error='分割引擎无法启动：'+self.process.errorString();self.on_finished(-1)
    def cancel(self):
        if self.isRunning():self.cancelled=True;self.process.kill()
    def on_finished(self,code,*_):
        if self.ended:return
        self.ended=True;self.read_output();self.read_error();out=Path(self.config['output'])
        log_path=out/'engine.log' if out.exists() else out.with_suffix('.log')
        log_path.write_text('\n'.join(self.log),encoding='utf8')
        if self.cancelled:
            if out.exists():(out/'cancelled.json').write_text(json.dumps({'cancelled':True,'partial_results_only':not (out/'complete.json').exists()}),encoding='utf8')
            self.progress.emit(0,'处理已取消，当前编辑工程未替换')
        elif code==0 and self.final and (out/'complete.json').exists():self.result.emit(self.final)
        else:self.failed.emit((self.error or f'分割引擎退出，错误码 {code}')+'\n日志：'+str(log_path))
        self.finished.emit()
