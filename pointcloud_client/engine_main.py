"""Isolated CPU inference process, newline-JSON progress, atomic job outcome."""
from pathlib import Path
import sys,os,json,time,traceback,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parent
if not getattr(sys,'frozen',False):sys.path.insert(0,str(ROOT.parent))

def emit(kind,**payload):print(json.dumps({'event':kind,**payload},ensure_ascii=True),flush=True)

def remesh(config):
    from core import Session
    from mesh_reconstruction import generate_meshes
    from exporter import export_3dm
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    try:
        session=Session.load(config['input']);started=time.perf_counter()
        def progress(i,n,name):emit('progress',percent=5+int(78*i/max(n,1)),message=f'贴合网格 {i+1}/{n}：{name}')
        geometry=generate_meshes(session,float(config.get('mesh_spacing',.02)),float(config.get('mesh_max_edge',.08)),config.get('group_ids'),progress)
        if geometry['generated']==0:raise ValueError('选中组未生成可靠网格，请调整采样间距与最大连边长度，或检查点云密度。'+str(geometry['skipped'][:3]))
        report={'geometry':geometry,'input_project':config['input'],'elapsed_seconds':time.perf_counter()-started}
        session.metadata['mesh_reconstruction']=geometry;session.save(out/'review.pcproj')
        emit('progress',percent=90,message='保存网格 3DM 并回读检查')
        report['export']=export_3dm(session,out/'model.3dm',True)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        (out/'complete.json').write_text(json.dumps({'success':True,'project':str(out/'review.pcproj'),'model':str(out/'model.3dm')}),encoding='utf8')
        emit('complete',project=str(out/'review.pcproj'),model=str(out/'model.3dm'),report=report)
    except Exception:
        (out/'failed.txt').write_text(traceback.format_exc(),encoding='utf8');raise

def run(config):
    if config.get('task')=='remesh':return remesh(config)
    import numpy as np
    from scan_io import read_scan
    from segmentation_pipeline import load_model,predict_semantics,save_ply,PALETTE
    from semantic_groups import split_semantic_groups
    from core import Session
    from auto_geometry import generate_candidates
    from exporter import export_3dm
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    try:
        started=time.perf_counter();emit('progress',percent=2,message='读取扫描、清理无效点')
        xyz,network,rgb,input_meta=read_scan(config['input'],float(config.get('scale',1)),config.get('up_axis','z'),bool(config.get('neutral',False)))
        digest=hashlib.sha256()
        with Path(config['input']).open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
        input_meta['source_sha256']=digest.hexdigest()
        checkpoint=Path(getattr(sys,'_MEIPASS',ROOT))/'weights/randlanet_s3dis.pth' if getattr(sys,'frozen',False) else ROOT.parent/'weights/randlanet_s3dis.pth'
        emit('progress',percent=8,message='加载室内建筑预训练模型（CPU）')
        model,model_meta=load_model(checkpoint,device='cpu',threads=min(os.cpu_count() or 4,8))
        def coverage(passes,fraction):emit('progress',percent=10+int(fraction*54),message=f'语义分割：第 {passes} 轮，覆盖 {fraction:.0%}')
        labels,confidence,coarse,pred_meta=predict_semantics(network,rgb,model,max_passes=300,progress=coverage)
        emit('progress',percent=67,message='按语义类别进行 RANSAC 平面与空间聚类分组')
        coarse_groups,manifest=split_semantic_groups(coarse['xyz'],coarse['semantic'],coarse['confidence'])
        groups=coarse_groups[coarse['inverse']]
        emit('progress',percent=78,message='保存完整分割点云')
        np.savez_compressed(out/'segmentation.npz',xyz=xyz,rgb=rgb.astype(np.uint8),semantic=labels,confidence=confidence,group_id=groups)
        save_ply(out/'semantic.ply',xyz,PALETTE[labels],labels,confidence,groups)
        report={'input':input_meta,'model':model_meta,'prediction':pred_meta,'groups':manifest,'room':Path(config['input']).stem,'grouping_coordinate_system':'Local inference coordinates in meters; add input.inference_origin_m for canonical coordinates','grouping_note':'RANSAC + DBSCAN candidates within predicted classes, not learned instances','group_count':len(manifest),'unassigned_group_points':int(np.sum(groups<0))}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        session=Session.load(out/'segmentation.npz')
        if config.get('auto_model',True):
            if config.get('reconstruction','mesh')=='mesh':
                from mesh_reconstruction import generate_meshes
                def meshing(i,n,name):emit('progress',percent=80+int(9*i/max(n,1)),message=f'贴合网格 {i+1}/{n}：{name}')
                report['geometry']=generate_meshes(session,float(config.get('mesh_spacing',.02)),float(config.get('mesh_max_edge',.08)),progress=meshing)
            else:
                emit('progress',percent=82,message='生成简化板体；不满足条件的组保留点云')
                report['geometry']=generate_candidates(session,float(config.get('thickness',.2)),float(config.get('tolerance',.04)),bool(config.get('boxes',False)))
        session.metadata['scan_job']={'raw_input':config['input'],'settings':config,'geometry':report.get('geometry')}
        session.save(out/'review.pcproj')
        emit('progress',percent=91,message='导出 Rhino 3DM 并回读验证')
        report['export']=export_3dm(session,out/'model.3dm',True)
        report['elapsed_seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        (out/'complete.json').write_text(json.dumps({'success':True,'points':len(xyz),'model':str(out/'model.3dm'),'project':str(out/'review.pcproj')},indent=2),encoding='utf8')
        emit('complete',project=str(out/'review.pcproj'),model=str(out/'model.3dm'),report=report)
    except Exception:
        (out/'failed.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    try:run(json.loads(args.config.read_text(encoding='utf8')))
    except Exception as e:emit('error',message=str(e));sys.exit(1)
