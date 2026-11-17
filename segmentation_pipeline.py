"""Portable pretrained S3DIS semantic segmentation and candidate grouping.

Uses the official Open3D-ML RandLA-Net layers/weights. SciPy kNN and NumPy
voxel averaging replace binary-dependent data operators. No Rhino dependency.
"""
from pathlib import Path
import argparse,hashlib,json,time,colorsys
import numpy as np
from scipy.spatial import cKDTree
import torch
from randlanet_portable import RandLANet

ROOT=Path(__file__).parent
CLASS_NAMES=['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
PALETTE=np.array([[233,180,65],[77,176,97],[75,135,210],[192,125,65],[159,105,196],[83,201,216],[221,111,92],[176,135,81],[216,104,174],[119,172,160],[128,107,204],[198,197,89],[140,140,140]],dtype=np.uint8)

def read_input(path,color_policy='require',scale=1.0,up_axis='z'):
    path=Path(path)
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as d:
            xyz=np.array(d['xyz'],dtype=np.float64)
            rgb=np.array(d['rgb']) if 'rgb' in d else None
    elif path.suffix.lower() in ['.pcd','.ply','.xyz','.xyzrgb','.pts']:
        import open3d as o3d
        p=o3d.io.read_point_cloud(str(path));xyz=np.asarray(p.points).copy()
        rgb=np.asarray(p.colors)*255.0 if p.has_colors() else None
    else:raise ValueError('Supported inputs: .npz (xyz/rgb), .pcd, .ply, .xyz, .xyzrgb, .pts')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<32:raise ValueError('Need an Nx3 point cloud with at least 32 points')
    if not np.isfinite(xyz).all():raise ValueError('Non-finite coordinates found; clean them explicitly before inference')
    missing=rgb is None
    if missing:
        if color_policy=='require':raise ValueError('This pretrained S3DIS model requires RGB. Supply colored data or explicitly use --color-policy neutral for an unvalidated fallback.')
        rgb=np.full(xyz.shape,127.5,np.float32)
    if rgb.shape!=xyz.shape or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:raise ValueError('RGB must have shape Nx3 and values in [0,255]')
    if not np.isfinite(scale) or scale<=0:raise ValueError('Scale must be finite and positive')
    # Preserve source coordinates for all exports. Rotate/scale only network input.
    rotation={'z':np.eye(3),'y':np.array([[1,0,0],[0,0,-1],[0,1,0]]),'x':np.array([[0,0,-1],[0,1,0],[1,0,0]])}[up_axis]
    network_xyz=xyz@rotation.T*scale
    return xyz,network_xyz.astype(np.float32),rgb.astype(np.float32),{'source':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scale_to_meters':scale,'up_axis':up_axis,'network_rotation':rotation.tolist(),'missing_rgb_fallback':missing,'point_count':len(xyz)}

def voxel_average(xyz,rgb,grid=.04):
    key=np.floor(xyz.astype(np.float64)/grid).astype(np.int64)
    _,inverse,count=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    p=np.stack([np.bincount(inverse,weights=xyz[:,j])/count for j in range(3)],1).astype(np.float32)
    c=np.stack([np.bincount(inverse,weights=rgb[:,j])/count for j in range(3)],1).astype(np.float32)
    return p,c,inverse

def load_model(checkpoint=None,device='auto',threads=8):
    checkpoint=Path(checkpoint or ROOT/'weights/randlanet_s3dis.pth')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is not available in this PyTorch environment')
    torch.set_num_threads(threads)
    model=RandLANet(num_layers=5,num_points=40960,num_classes=13,ignored_label_inds=[],sub_sampling_ratio=[4,4,4,4,2],in_channels=6,dim_features=8,dim_output=[16,64,128,256,512],grid_size=.04)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.to(device).eval()
    return model,{'checkpoint':checkpoint.name,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'device':device,'torch':torch.__version__,'parameters':sum(p.numel() for p in model.parameters())}

def patch_inputs(points,rgb):
    p=points.copy();p[:,:2]-=p[:,:2].mean(0)
    features=np.concatenate([p,rgb],axis=1)
    coords=[];neighbors=[];pools=[];interpolation=[]
    for ratio in [4,4,4,4,2]:
        k=min(16,len(p));index=cKDTree(p).query(p,k=k,workers=-1)[1]
        if index.ndim==1:index=index[:,None]
        if k<16:index=np.pad(index,((0,0),(0,16-k)),mode='edge')
        sub=p[:max(1,len(p)//ratio)]
        interp=cKDTree(sub).query(p,k=1,workers=-1)[1][:,None]
        coords.append(torch.from_numpy(p[None].copy()))
        neighbors.append(torch.from_numpy(index[None].astype(np.int64)))
        pools.append(torch.from_numpy(index[None,:len(sub)].astype(np.int64)))
        interpolation.append(torch.from_numpy(interp[None].astype(np.int64)))
        p=sub
    return {'features':torch.from_numpy(features[None]),'coords':coords,'neighbor_indices':neighbors,'sub_idx':pools,'interp_idx':interpolation}

def predict_semantics(xyz,rgb,model,grid=.04,patch_size=40960,min_possibility=.5,max_passes=100,seed=42,progress=True):
    if not np.isfinite(grid) or grid<=0 or patch_size<4096:raise ValueError('grid must be finite and positive and patch_size >= 4096')
    if not np.isfinite(min_possibility) or min_possibility<=0 or max_passes<1:raise ValueError('Invalid coverage settings')
    start=time.perf_counter();p,c,inverse=voxel_average(xyz,rgb,grid)
    rng=np.random.default_rng(seed);tree=cKDTree(p)
    possibility=rng.uniform(0,.001,len(p));scores=np.zeros((len(p),13),np.float32);mass=np.zeros(len(p),np.float32);visits=np.zeros(len(p),np.uint16)
    with torch.inference_mode():
        for pass_no in range(max_passes):
            center=int(np.argmin(possibility))
            if len(p)>=patch_size:index=tree.query(p[center],k=patch_size)[1]
            else:index=np.r_[np.arange(len(p)),rng.choice(len(p),patch_size-len(p),replace=True)]
            rng.shuffle(index)
            inputs=patch_inputs(p[index],c[index])
            prob=torch.softmax(model(inputs)[0],dim=-1).cpu().numpy()
            if not np.isfinite(prob).all():raise RuntimeError('Non-finite model predictions')
            unique,first=np.unique(index,return_index=True)
            scores[unique]=.95*scores[unique]+.05*prob[first]
            mass[unique]=.95*mass[unique]+.05;visits[unique]+=1
            dist=((p[unique]-p[center])**2).sum(1)
            delta=(1-dist/max(float(dist.max()),1e-10))**2
            possibility[unique]+=delta
            if callable(progress):progress(pass_no+1,float((possibility>min_possibility).mean()))
            elif progress:print(f'pass {pass_no+1}: visited {(visits>0).mean():.1%}, regular coverage {(possibility>min_possibility).mean():.1%}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
            if possibility.min()>min_possibility:break
        else:raise RuntimeError('Coverage incomplete at max_passes; no final output emitted. Increase max_passes or patch_size.')
    scores/=mass[:,None]
    labels=scores.argmax(1).astype(np.uint8);confidence=scores.max(1)
    meta={'coarse_points':len(p),'passes':pass_no+1,'grid_size':grid,'patch_size':patch_size,'seed':seed,'coverage_min':float(possibility.min()),'minimum_visits':int(visits.min()),'inference_seconds':time.perf_counter()-start,'sampling':'spatially regular nearest-neighbor crops, original 0.95 EMA voting with bias correction','adaptations':'NumPy voxel averages and SciPy kNN; neural forward graph and weights unchanged','confidence_note':'Maximum softmax score, not calibrated probability of correctness.'}
    return labels[inverse],confidence[inverse],{'xyz':p,'rgb':c,'semantic':labels,'confidence':confidence,'scores':scores,'inverse':inverse},meta

def evaluate_labels(truth,pred):
    truth=np.asarray(truth).astype(np.int64);pred=np.asarray(pred).astype(np.int64)
    if len(truth)!=len(pred) or np.any((truth<0)|(truth>=13)) or np.any((pred<0)|(pred>=13)):raise ValueError('Invalid labels for evaluation')
    cm=np.bincount(truth*13+pred,minlength=169).reshape(13,13)
    union=cm.sum(0)+cm.sum(1)-cm.diagonal();valid=union>0
    iou=np.divide(cm.diagonal(),union,out=np.full(13,np.nan),where=valid)
    return {'points':len(truth),'accuracy':float(np.trace(cm)/cm.sum()),'miou_present_union':float(np.nanmean(iou)),'class_iou':{n:(float(iou[i]) if valid[i] else None) for i,n in enumerate(CLASS_NAMES)},'truth_counts':dict(zip(CLASS_NAMES,cm.sum(1).tolist())),'predicted_counts':dict(zip(CLASS_NAMES,cm.sum(0).tolist())),'confusion_matrix':cm.tolist()}

def save_ply(path,xyz,color,semantic,confidence,group_id):
    """Binary PLY retains per-point class, confidence and group properties."""
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('semantic','u1'),('confidence','<f4'),('group_id','<i4')])
    a=np.empty(len(xyz),dtype=dtype)
    for i,k in enumerate(['x','y','z']):a[k]=xyz[:,i]
    for i,k in enumerate(['red','green','blue']):a[k]=color[:,i]
    a['semantic']=semantic;a['confidence']=confidence;a['group_id']=group_id
    header='ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(xyz)}\n'+'\n'.join(['property float x','property float y','property float z','property uchar red','property uchar green','property uchar blue','property uchar semantic','property float confidence','property int group_id'])+'\nend_header\n'
    with Path(path).open('wb') as f:f.write(header.encode('ascii'));f.write(a.tobytes())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--color-policy',choices=['require','neutral'],default='require');parser.add_argument('--scale-to-meters',type=float,default=1.)
    parser.add_argument('--up-axis',choices=['x','y','z'],default='z');parser.add_argument('--grid',type=float,default=.04)
    parser.add_argument('--patch-size',type=int,default=40960);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-passes',type=int,default=100)
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--skip-groups',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Output directory is nonempty; choose a new result directory to preserve prior experiments.')
    source,network,rgb,input_meta=read_input(args.input,args.color_policy,args.scale_to_meters,args.up_axis)
    model,model_meta=load_model(device=args.device,threads=args.threads)
    print('Loaded official checkpoint strictly:',model_meta,flush=True)
    labels,confidence,coarse,pred_meta=predict_semantics(network,rgb,model,args.grid,args.patch_size,max_passes=args.max_passes,seed=args.seed)
    groups=np.full(len(source),-1,np.int32);group_manifest=[]
    if not args.skip_groups:
        from semantic_groups import split_semantic_groups
        coarse_groups,group_manifest=split_semantic_groups(coarse['xyz'],coarse['semantic'],coarse['confidence'],seed=args.seed)
        groups=coarse_groups[coarse['inverse']]
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'segmentation.npz',xyz=source,rgb=rgb.astype(np.uint8),semantic=labels,confidence=confidence,group_id=groups)
    np.savez_compressed(args.output/'coarse_predictions.npz',**coarse)
    save_ply(args.output/'semantic.ply',source,PALETTE[labels],labels,confidence,groups)
    group_colors=np.full((len(source),3),145,np.uint8)
    for group in group_manifest:
        i=group['id'];color=np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.65,.92))*255
        group_colors[groups==i]=color.astype(np.uint8);group['original_points']=int(sum(groups==i))
    save_ply(args.output/'groups.ply',source,group_colors,labels,confidence,groups)
    report={'input':input_meta,'model':model_meta,'prediction':pred_meta,'classes':CLASS_NAMES,'groups':group_manifest,'group_count':len(group_manifest),'unassigned_group_points':int(sum(groups<0)),
        'grouping_note':'Geometric candidates, not learned instances. Small fragments keep group_id=-1. Every point retains its semantic prediction. Group bounds/planes are in rotated/scaled network coordinates; NPZ/PLY points retain source coordinates.'}
    if not args.skip_groups:report['grouping_parameters']={'eps_m':.14,'minimum_voxels':35,'plane_distance_m':.035,'ransac_trials':500,'seed':args.seed,'implementation':'NumPy fixed-order RANSAC + DBSCAN'}
    if args.evaluate:
        if args.input.suffix!='.npz':raise ValueError('--evaluate requires NPZ with ground truth semantic field')
        with np.load(args.input,allow_pickle=False) as d:truth=d['semantic'].copy()
        report['evaluation']=evaluate_labels(truth,labels)
        report['evaluation']['scope']='Single supplied room only; not full S3DIS benchmark. Ground truth is loaded only after predictions.'
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k in ['group_count','unassigned_group_points','evaluation']},indent=2),flush=True)
    print('Saved',args.output,flush=True)

if __name__=='__main__':main()
