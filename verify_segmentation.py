"""Integration checks, including real RGB PLY with ALL labels removed."""
from pathlib import Path
import json,tempfile,time
import numpy as np
import open3d as o3d
from segmentation_pipeline import read_input,load_model,predict_semantics,evaluate_labels
from semantic_groups import split_semantic_groups
ROOT=Path(__file__).parent

def verify(root=ROOT):
    # Metrics scope is descriptive metadata; compare numeric result separately.
    start=time.perf_counter();checks={}
    with np.load(root/'data/processed/Area_3_office_1.npz') as d:
        xyz=d['xyz'].copy();rgb=d['rgb'].copy();truth=d['semantic'].copy()
    with np.load(root/'results/pretrained_area3/segmentation.npz') as d:saved={k:d[k].copy() for k in d.files}
    with np.load(root/'results/pretrained_area3/coarse_predictions.npz') as d:coarse={k:d[k].copy() for k in d.files}
    report=json.loads((root/'results/pretrained_area3/report.json').read_text())
    assert np.array_equal(saved['xyz'],xyz) and np.array_equal(saved['rgb'],rgb)
    assert len(saved['semantic'])==len(xyz)
    assert np.isfinite(saved['confidence']).all() and np.all((saved['confidence']>=0)&(saved['confidence']<=1.000001))
    assert np.allclose(coarse['scores'].sum(1),1,atol=2e-6)
    assert np.array_equal(saved['semantic'],coarse['semantic'][coarse['inverse']])
    checks['point_count_coordinates_rgb_probabilities_and_voxel_mapping']='PASS'
    for gid in set(saved['group_id'])-{-1}:assert len(np.unique(saved['semantic'][saved['group_id']==gid]))==1
    assert sum(g['original_points'] for g in report['groups'])+report['unassigned_group_points']==len(xyz)
    checks['group_semantic_consistency_and_point_conservation']='PASS'
    regrouped,regrouped_manifest=split_semantic_groups(coarse['xyz'],coarse['semantic'],coarse['confidence'])
    assert np.array_equal(regrouped[coarse['inverse']],saved['group_id'])
    checks['fixed_seed_grouping_reproduces_saved_point_assignments']='PASS'
    measured=evaluate_labels(truth,saved['semantic'])
    for k,v in measured.items():assert report['evaluation'][k]==v
    checks['independently_recomputed_confusion_matrix_and_metrics']='PASS'
    with tempfile.TemporaryDirectory(prefix='pointcloud_test_') as temporary:
        temp=Path(temporary)
        p=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz));p.colors=o3d.utility.Vector3dVector(rgb/255.)
        assert o3d.io.write_point_cloud(str(temp/'unlabeled.ply'),p)
        source,network,color,meta=read_input(temp/'unlabeled.ply')
        assert np.array_equal(source,xyz) and np.allclose(color,rgb)
        model,_=load_model()
        labels,confidence,_,prediction_meta=predict_semantics(network,color,model,progress=False)
        assert np.array_equal(labels,saved['semantic'])
        assert np.allclose(confidence,saved['confidence'],atol=2e-6)
        checks['unlabeled_rgb_ply_same_predictions_as_npz']='PASS: full room; all ground-truth fields removed'
        assert o3d.io.write_point_cloud(str(temp/'colored.pcd'),p,compressed=True)
        pcd_source,pcd_network,pcd_rgb,_=read_input(temp/'colored.pcd')
        assert np.array_equal(pcd_source,xyz) and np.allclose(pcd_rgb,rgb)
        checks['compressed_rgb_pcd_roundtrip']='PASS'
        p.colors=o3d.utility.Vector3dVector()
        assert o3d.io.write_point_cloud(str(temp/'no_rgb.pcd'),p)
        try:read_input(temp/'no_rgb.pcd')
        except ValueError as e:assert 'requires RGB' in str(e)
        else:raise AssertionError('Missing RGB accepted silently')
        checks['missing_rgb_rejected_by_default']='PASS'
        tiny=xyz[:100].copy();tiny[0,0]=np.nan
        np.savez(temp/'invalid.npz',xyz=tiny,rgb=rgb[:100])
        try:read_input(temp/'invalid.npz')
        except ValueError as e:assert 'Non-finite' in str(e)
        else:raise AssertionError('NaN coordinates accepted')
        checks['invalid_coordinates_rejected']='PASS'
        np.savez(temp/'millimeters_y_up.npz',xyz=xyz[:100]*1000,rgb=rgb[:100])
        original,network,_,_=read_input(temp/'millimeters_y_up.npz',scale=.001,up_axis='y')
        assert np.array_equal(original,xyz[:100]*1000)
        assert np.allclose(network[:,2],xyz[:100,1],atol=1e-5)
        checks['scale_and_up_axis_keep_export_coordinates']='PASS'
    exported=o3d.io.read_point_cloud(str(root/'results/pretrained_area3/semantic.ply'))
    assert np.array_equal(np.asarray(exported.points),xyz)
    checks['binary_ply_point_count_and_coordinate_roundtrip']='PASS'
    # Synthetic two distant planar walls, no annotated source instances.
    a,b=np.meshgrid(np.arange(0,1,.04),np.arange(0,1,.04))
    points=np.r_[np.c_[a.ravel(),a.ravel()*0,b.ravel()],np.c_[a.ravel()+3,a.ravel()*0,b.ravel()]]
    ids,groups=split_semantic_groups(points,np.full(len(points),2),np.ones(len(points)))
    assert len(groups)==2 and np.all(ids>=0) and len(set(ids[:len(a.ravel())]))==1
    checks['disconnected_coplanar_walls_split_into_two_candidates']='PASS'
    result={'checks':checks,'passed':len(checks),'seconds':time.perf_counter()-start,
            'scope':'Software checks and one-room repeatability; not proof of generalization or calibrated confidence.'}
    (root/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':verify()
