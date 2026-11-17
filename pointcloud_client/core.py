"""Offline point-cloud review model. No Rhino process, network, or ML dependency."""
from pathlib import Path
import copy,json,os,colorsys
from datetime import datetime,timezone
import numpy as np

CLASS_NAMES=['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
CLASS_ZH=['天花板','地面','墙','梁','柱','窗','门','桌子','椅子','沙发','书架','板面','杂项']
PALETTE=np.array([[233,180,65],[77,176,97],[75,135,210],[192,125,65],[159,105,196],[83,201,216],[221,111,92],[176,135,81],[216,104,174],[119,172,160],[128,107,204],[198,197,89],[140,140,140]],np.uint8)

def group_color(gid):
    return np.array(colorsys.hsv_to_rgb((gid*.61803398875)%1,.62,.95)) if gid>=0 else np.array([.58,.62,.67])

class Session:
    def __init__(self,xyz,rgb,prediction,confidence,assignment,info=None,models=None,metadata=None):
        raw_rgb=np.asarray(rgb)
        if not np.isfinite(raw_rgb).all() or np.any((raw_rgb<0)|(raw_rgb>255)):raise ValueError('RGB必须在0到255之间。')
        self.xyz=np.asarray(xyz,dtype=np.float64);self.rgb=np.asarray(rgb,dtype=np.uint8)
        self.prediction=np.asarray(prediction,dtype=np.int16);self.confidence=np.asarray(confidence,dtype=np.float32)
        self.assignment=np.asarray(assignment,dtype=np.int32).copy();n=len(self.xyz)
        if self.xyz.shape!=(n,3) or self.rgb.shape!=(n,3) or n<3:raise ValueError('点云必须包含至少3个点及对应RGB。')
        if not all(len(a)==n for a in [self.prediction,self.confidence,self.assignment]):raise ValueError('点与标签数量不一致。')
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.confidence).all():raise ValueError('输入含无效坐标或模型分数。')
        if np.any((self.prediction<0)|(self.prediction>12)) or np.any(self.assignment < -1):raise ValueError('类别或分组编号无效。')
        self.info={int(k):v for k,v in (info or {}).items()};self.models=models or [];self.metadata=metadata or {}
        for gid in np.unique(self.assignment):
            gid=int(gid)
            if gid not in self.info:
                values=self.prediction[self.assignment==gid];unique=np.unique(values)
                cid=int(unique[0]) if len(unique)==1 else -1
                self.info[gid]={'name':('待归组' if gid<0 else f'G{gid+1:03d}'),'class_id':cid,'visible':True}
        if set(self.info)!=set(map(int,np.unique(self.assignment))):raise ValueError('工程分组表与点编号不一致。')
        if any(v.get('class_id') not in range(-1,13) for v in self.info.values()):raise ValueError('工程包含无效类别。')
        for model in self.models:
            if model['group_id'] not in self.info:raise ValueError('几何引用了不存在的分组')
            if model['kind']=='mesh':
                p=np.asarray(model['vertices']);f=np.asarray(model['faces']);c=np.asarray(model['colors'])
                if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():raise ValueError('工程网格顶点无效')
                if f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(p):raise ValueError('工程网格面索引无效')
                if c.shape!=p.shape or not np.isfinite(c).all() or c.min()<0 or c.max()>255:raise ValueError('工程网格颜色无效')
        self.undo_stack=[];self.redo_stack=[];self.dirty=False

    @classmethod
    def load(cls,path):
        path=Path(path)
        with np.load(path,allow_pickle=False) as d:
            required={'xyz','rgb','semantic','confidence','group_id'}
            if not required.issubset(d.files):raise ValueError('请选择分割结果 segmentation.npz 或本客户端保存的 .pcproj。')
            arrays={k:d[k].copy() for k in required}
            saved=json.loads(str(d['client_metadata'].item())) if 'client_metadata' in d else None
            if saved and saved.get('version')==2:
                for i,model in enumerate(saved['models']):
                    if model['kind']=='mesh':
                        for field in ['vertices','faces','colors']:model[field]=d[f'mesh_{i}_{field}'].copy()
        if saved:
            if saved.get('version') not in [1,2]:raise ValueError('不支持此工程文件版本。')
            return cls(arrays['xyz'],arrays['rgb'],arrays['semantic'],arrays['confidence'],arrays['group_id'],saved['groups'],saved['models'],saved['metadata'])
        report_path=path.with_name('report.json');report=json.loads(report_path.read_text(encoding='utf8')) if report_path.exists() else {}
        source=report.get('input',{});scale=float(source.get('scale_to_meters',1.));rotation=np.array(source.get('network_rotation',np.eye(3)),float)
        if not np.isfinite(scale) or scale<=0 or rotation.shape!=(3,3) or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-5):raise ValueError('结果文件单位/轴向信息无效。')
        xyz=np.asarray(arrays['xyz'],float)@rotation.T*scale
        meta={'source_file':str(path.resolve()),'room':report.get('room',path.parent.name),'scale_to_meters':scale,'source_rotation':rotation.tolist(),'coordinate_policy':'Canonical meters, rotated/scaled from source; no translation applied.','audit':[]}
        return cls(xyz,arrays['rgb'],arrays['semantic'],arrays['confidence'],arrays['group_id'],metadata=meta)

    def snapshot(self):return (self.assignment.copy(),copy.deepcopy(self.info),copy.deepcopy(self.models),copy.deepcopy(self.metadata))
    def checkpoint(self,action):
        self.undo_stack.append(self.snapshot());self.undo_stack=self.undo_stack[-20:];self.redo_stack=[];self.dirty=True
        self.metadata.setdefault('audit',[]).append({'action':action,'utc':datetime.now(timezone.utc).isoformat()})
    def undo(self,redo=False):
        source=self.redo_stack if redo else self.undo_stack;target=self.undo_stack if redo else self.redo_stack
        if not source:return False
        target.append(self.snapshot());self.assignment,self.info,self.models,self.metadata=source.pop();self.dirty=True;return True
    def indices(self,gids):return np.flatnonzero(np.isin(self.assignment,list(gids)))
    def invalidate(self,gids):self.models=[m for m in self.models if m['group_id'] not in gids]
    def next_id(self):return max([0]+list(self.info))+1
    def merge(self,gids,cid):
        gids=set(map(int,gids))
        if len(gids)<2 or not gids.issubset(self.info):raise ValueError('请至少选择两个分组。')
        if cid not in range(-1,13):raise ValueError('请选择合并后的类别。')
        mask=np.isin(self.assignment,list(gids));new=self.next_id()
        self.checkpoint('merge '+str(sorted(gids)));self.invalidate(gids);self.assignment[mask]=new
        for gid in gids:del self.info[gid]
        self.info[new]={'name':f'合并组 {new+1:03d}','class_id':cid,'visible':True};return new
    def split(self,gid,axis,position):
        if gid not in self.info or axis not in [0,1,2] or not np.isfinite(position):raise ValueError('请选择有效分组与切分平面。')
        members=self.assignment==gid;right=members&(self.xyz[:,axis]>position);left=members&~right
        if right.sum()<3 or left.sum()<3:raise ValueError('切分后两侧都需至少3个点，请调整切分位置。')
        new=self.next_id();self.checkpoint(f'split {gid} axis{axis} at {position}');self.invalidate({gid})
        self.assignment[right]=new;info=copy.deepcopy(self.info[gid]);base=info['name']
        self.info[gid]['name']=base+' A';info['name']=base+' B';self.info[new]=info;return gid,new
    def rename(self,gid,name,cid):
        if not name.strip():raise ValueError('组名不能为空。')
        if cid not in range(-1,13):raise ValueError('类别无效。')
        self.checkpoint(f'edit group {gid}');self.info[gid].update(name=name.strip(),class_id=cid)
        for model in self.models:
            if model['group_id']==gid:model['name']=name.strip()
    def clear_models(self):
        if self.models:self.checkpoint('clear generated models');self.models=[]
    def save(self,path):
        path=Path(path);tmp=path.with_name(path.name+'.tmp')
        models=[];mesh_arrays={}
        for i,model in enumerate(self.models):
            if model['kind']=='mesh':
                models.append({k:v for k,v in model.items() if k not in ['vertices','faces','colors']})
                for field in ['vertices','faces','colors']:mesh_arrays[f'mesh_{i}_{field}']=np.asarray(model[field])
            else:models.append(model)
        metadata={'version':2 if mesh_arrays else 1,'groups':self.info,'models':models,'metadata':self.metadata}
        with tmp.open('wb') as f:
            np.savez_compressed(f,xyz=self.xyz,rgb=self.rgb,semantic=self.prediction,confidence=self.confidence,group_id=self.assignment,client_metadata=np.array(json.dumps(metadata,ensure_ascii=False)),**mesh_arrays)
        os.replace(tmp,path);self.dirty=False

def fit_candidate(points,gid,name,kind='slab',thickness=.20,tolerance=.04):
    p=np.asarray(points,dtype=np.float64)
    if len(p)<10:raise ValueError('至少需要10个点进行拟合。')
    if not np.isfinite([thickness,tolerance]).all() or thickness<=0 or tolerance<=0:raise ValueError('厚度和允许偏差必须大于0。')
    sample=p[np.linspace(0,len(p)-1,min(len(p),50000),dtype=int)]
    center=np.median(sample,axis=0);work=sample
    for _ in range(3):
        center=work.mean(0);_,s,v=np.linalg.svd(work-center,full_matrices=False)
        normal=v[-1];distance=abs((sample-center)@normal)
        work=sample[distance<=max(np.quantile(distance,.9),1e-9)]
        if len(work)<10:break
    if normal[np.argmax(abs(normal))]<0:normal=-normal
    # Keep vertical architectural sheets upright; horizontal sheets follow dominant axis.
    if abs(normal[2])<.9:
        v_axis=np.array([0.,0.,1.]);v_axis-=normal*normal[2];v_axis/=np.linalg.norm(v_axis)
        u_axis=np.cross(v_axis,normal)
    else:
        u_axis=v[0]-normal*(v[0]@normal);u_axis/=np.linalg.norm(u_axis);v_axis=np.cross(normal,u_axis)
    basis=np.column_stack([u_axis,v_axis,normal]);local=(p-center)@basis
    rms=float(np.sqrt(np.mean(local[:,2]**2)));p95=float(np.quantile(abs(local[:,2]),.95))
    if kind in ['surface','slab'] and p95>tolerance:raise ValueError(f'95%点到拟合平面的距离为 {p95:.3f} m，超过允许偏差 {tolerance:.3f} m。请先拆组，或明确使用包围盒体块。')
    if kind=='box':
        center=sample.mean(0);_,_,v=np.linalg.svd(sample-center,full_matrices=False);basis=v.T
        if np.linalg.det(basis)<0:basis[:,2]*=-1
        local=(p-center)@basis
    low=local.min(0);high=local.max(0)
    if high[0]-low[0]<.001 or high[1]-low[1]<.001:raise ValueError('点集近似直线，无法生成有效面。')
    if kind=='slab':low[2]=-thickness/2;high[2]=thickness/2
    elif kind=='surface':low[2]=high[2]=0
    elif kind=='box':
        if high[2]-low[2]<.001:low[2]-=thickness/2;high[2]+=thickness/2
    else:raise ValueError('未知建模方式。')
    return {'group_id':int(gid),'name':name,'kind':kind,'center':center.tolist(),'basis':basis.tolist(),'low':low.tolist(),'high':high.tolist(),
        'thickness_m':float(thickness),'rms_m':rms,'p95_m':p95,'points':len(p),'assumption':'Rectangular envelope; openings are not reconstructed. Slab thickness is user supplied; box is an approximate envelope.'}

def model_vertices(model):
    if model['kind']=='mesh':return np.asarray(model['vertices'])+np.asarray(model['center'])
    lo=np.array(model['low']);hi=np.array(model['high'])
    corners=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    return corners@np.array(model['basis']).T+np.array(model['center'])
