"""Small supervised baseline: train on Area 1 office 1, evaluate Area 3 office 1.

Training data volume differs from pretrained RandLA-Net: not an architecture ablation.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from segmentation_pipeline import voxel_average,evaluate_labels,CLASS_NAMES
ROOT=Path(__file__).parent

def point_features(xyz,rgb):
    neighbors=cKDTree(xyz).query(xyz,k=min(20,len(xyz)),workers=-1)[1]
    patches=xyz[neighbors];centered=patches-patches.mean(1,keepdims=True)
    covariance=np.einsum('nki,nkj->nij',centered,centered)/patches.shape[1]
    values,vectors=np.linalg.eigh(covariance);values=np.maximum(values,0)
    normal=np.abs(vectors[:,:,0]);denom=np.maximum(values[:,2],1e-10)
    shape=np.stack([(values[:,2]-values[:,1])/denom,(values[:,1]-values[:,0])/denom,
                    values[:,0]/np.maximum(values.sum(1),1e-10)],1)
    # No absolute XY, room identity or instance IDs as features.
    return np.column_stack([xyz[:,2],rgb/255.,normal,shape]).astype(np.float32)

def run_baseline(root=ROOT):
    start=time.perf_counter()
    with np.load(root/'data/processed/Area_1_office_1.npz') as d:
        p,c,inverse=voxel_average(d['xyz'],d['rgb'])
        votes=np.bincount(inverse*13+d['semantic'],minlength=len(p)*13).reshape(-1,13)
        y=votes.argmax(1)
    train_features=point_features(p,c)
    model=RandomForestClassifier(n_estimators=100,max_depth=22,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=8)
    model.fit(train_features,y);train_seconds=time.perf_counter()-start
    with np.load(root/'data/processed/Area_3_office_1.npz') as d:
        test_xyz=d['xyz'];test_rgb=d['rgb'];truth=d['semantic']
    start=time.perf_counter();p,c,inverse=voxel_average(test_xyz,test_rgb)
    predicted=model.predict(point_features(p,c)).astype(np.uint8)[inverse]
    seconds=time.perf_counter()-start
    report={'method':'Random forest: height + RGB + local normal + eigenvalue shape features',
        'train_room':'Area_1/office_1','test_room':'Area_3/office_1','train_voxels':len(y),
        'train_seconds':train_seconds,'predict_seconds':seconds,'parameters':model.get_params(),
        'evaluation':evaluate_labels(truth,predicted),
        'comparison_limit':'One-room local training versus multi-room pretrained model; both use identical test points and 13-class mapping. This is a practical pipeline baseline, not a controlled architecture comparison.'}
    out=root/'results/random_forest';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',semantic=predicted)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('RF accuracy:',report['evaluation']['accuracy'],'mIoU:',report['evaluation']['miou_present_union'],flush=True)
    return report

if __name__=='__main__':run_baseline()
