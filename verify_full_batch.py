"""Read back all 23 saved predictions and recompute metrics from source truth."""
from pathlib import Path
import json
import numpy as np
from segmentation_pipeline import evaluate_labels
ROOT=Path(__file__).parent

def verify_batch(root=ROOT):
    data_manifest=json.loads((root/'data/full_processed/manifest.json').read_text())
    summary=json.loads((root/'results/area3_batch/summary.json').read_text())
    expected={r['room']:r for r in data_manifest['rooms'] if r['room'].startswith('Area_3/')}
    assert len(expected)==23 and {r['room'] for r in summary['rooms']}==set(expected)
    cm=np.zeros((13,13),np.int64);points=0
    for room in summary['rooms']:
        identifier=room['room'].replace('/','_');folder=root/'results/area3_batch'/identifier
        with np.load(root/expected[room['room']]['file']) as original,np.load(folder/'segmentation.npz') as output:
            assert np.array_equal(original['xyz'],output['xyz'])
            assert np.array_equal(original['rgb'],output['rgb'])
            count=len(original['xyz']);assert count==room['points']
            assert all(len(output[key])==count for key in ['semantic','confidence','group_id'])
            assert np.isfinite(output['confidence']).all()
            measured=evaluate_labels(original['semantic'],output['semantic'])
            report=json.loads((folder/'report.json').read_text())
            assert measured==report['evaluation']
            groups=output['group_id'];labels=output['semantic']
            assert np.all((groups>=-1)&(groups<report['group_count']))
            counts=np.bincount(groups[groups>=0],minlength=report['group_count'])
            for group in report['groups']:
                gid=group['id'];assert counts[gid]==group['original_points']
                assert np.all(labels[groups==gid]==group['class_id'])
            assert int((groups<0).sum())==report['unassigned_group_points']
            cm+=np.asarray(measured['confusion_matrix'],np.int64);points+=count
        assert (folder/'semantic.ply').exists() and (folder/'groups.ply').exists()
    assert cm.tolist()==summary['aggregate']['confusion_matrix'] and points==summary['aggregate']['points']
    checks={'rooms_checked':23,'points_checked':points,'all_saved_xyz_and_rgb_equal_source':True,
        'all_room_metrics_recomputed_from_saved_predictions':True,'all_group_class_ids_and_point_counts_consistent':True,
        'aggregate_confusion_matrix_matches':True,'failed_checks':0}
    (root/'results/area3_batch/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2),flush=True);return checks

if __name__=='__main__':verify_batch()
