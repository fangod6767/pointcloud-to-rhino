"""Frozen pretrained inference across all 23 Area 3 rooms; no retraining/tuning."""
from pathlib import Path
import json,time,hashlib,colorsys,traceback
import numpy as np
from segmentation_pipeline import read_input,load_model,predict_semantics,evaluate_labels,save_ply,PALETTE,CLASS_NAMES
from semantic_groups import split_semantic_groups
from prepare_full_s3dis import save_npz_fast
ROOT=Path(__file__).parent

def metrics_from_confusion(cm):
    cm=np.asarray(cm,dtype=np.int64);union=cm.sum(0)+cm.sum(1)-np.diag(cm);valid=union>0
    iou=np.divide(np.diag(cm),union,out=np.zeros(13,float),where=valid)
    return {'points':int(cm.sum()),'accuracy':float(np.trace(cm)/cm.sum()),'miou_present_union':float(iou[valid].mean()),
        'class_iou':{name:float(iou[i]) if valid[i] else None for i,name in enumerate(CLASS_NAMES)},'confusion_matrix':cm.tolist()}

def batch(root=ROOT):
    manifest_path=root/'data/full_processed/manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        rooms=[r for r in manifest['rooms'] if r['room'].startswith('Area_3/')]
    else:
        # Other areas may still be converting; each room JSON is written only after its NPZ is complete.
        inventory=json.loads((root/'data/full_s3dis_inventory.json').read_text())
        rooms=[json.loads((root/'data/full_processed'/('Area_3_'+name+'.json')).read_text()) for name in inventory['areas']['Area_3']]
    assert len(rooms)==23
    out=root/'results/area3_batch';out.mkdir(parents=True,exist_ok=True)
    model,model_meta=load_model();reports=[];failures=[];start=time.perf_counter()
    code_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['segmentation_pipeline.py','semantic_groups.py','randlanet_portable.py']}
    for index,item in enumerate(rooms):
        name=item['room'].replace('/','_');dest=out/name;report_path=dest/'report.json'
        if report_path.exists():
            prior=json.loads(report_path.read_text())
            if prior.get('code_sha256')!=code_hashes:raise ValueError('Prior result has different code; use a new output root')
            reports.append(prior);print('Reuse verified completed room',name,flush=True);continue
        print(f"START {index+1}/{len(rooms)} {name} {item['points']:,} points",flush=True)
        try:
            source,network,rgb,input_meta=read_input(root/item['file'])
            labels,confidence,coarse,prediction=predict_semantics(network,rgb,model,max_passes=1000,progress=False)
            ids,groups=split_semantic_groups(coarse['xyz'],coarse['semantic'],coarse['confidence'])
            full_ids=ids[coarse['inverse']]
            with np.load(root/item['file']) as annotated:truth=annotated['semantic'].copy()
            evaluation=evaluate_labels(truth,labels)
            assert len(labels)==len(source) and np.isfinite(confidence).all()
            color=np.full((len(source),3),145,np.uint8)
            counts=np.bincount(full_ids[full_ids>=0],minlength=len(groups))
            for group in groups:
                g=group['id'];group['original_points']=int(counts[g])
                color[full_ids==g]=np.array(colorsys.hsv_to_rgb((g*.61803398875)%1,.65,.92))*255
            assert sum(counts)+int((full_ids<0).sum())==len(source)
            dest.mkdir(parents=True,exist_ok=True)
            save_npz_fast(dest/'segmentation.npz',xyz=source,rgb=rgb.astype(np.uint8),semantic=labels,confidence=confidence,group_id=full_ids)
            save_ply(dest/'semantic.ply',source,PALETTE[labels],labels,confidence,full_ids)
            save_ply(dest/'groups.ply',source,color,labels,confidence,full_ids)
            report={'room':item['room'],'input':input_meta,'model':model_meta,'code_sha256':code_hashes,'prediction':prediction,
                'classes':CLASS_NAMES,'groups':groups,'group_count':len(groups),'unassigned_group_points':int((full_ids<0).sum()),
                'evaluation':evaluation,'grouping_parameters':{'eps_m':.14,'minimum_voxels':35,'plane_distance_m':.035,'ransac_trials':500,'seed':42},
                'grouping_note':'Geometric candidates only, not measured instance accuracy. Group bounds and planes use network coordinates in meters.'}
            report_path.write_text(json.dumps(report,indent=2),encoding='utf8');reports.append(report)
            print(f"DONE {name} accuracy={evaluation['accuracy']:.2%} mIoU={evaluation['miou_present_union']:.2%} groups={len(groups)} inference={prediction['inference_seconds']:.1f}s",flush=True)
        except Exception as exc:
            failures.append({'room':item['room'],'error':str(exc),'traceback':traceback.format_exc()})
            print('FAILED',name,str(exc),flush=True)
        checkpoint={'completed_rooms':[r['room'] for r in reports],'failures':failures,'target_rooms':len(rooms)}
        (out/'progress.json').write_text(json.dumps(checkpoint,indent=2),encoding='utf8')
    cm=sum((np.asarray(r['evaluation']['confusion_matrix'],dtype=np.int64) for r in reports),np.zeros((13,13),np.int64))
    summary={'dataset':'S3DIS v1.2 Aligned Version','area':'Area_3','expected_rooms':23,'completed_rooms':len(reports),'failures':failures,
        'aggregate':metrics_from_confusion(cm),'mean_room_miou':float(np.mean([r['evaluation']['miou_present_union'] for r in reports])),
        'scope':'All available Area 3 rooms; official model config selects area 3 for testing. No local training or parameter selection on these labels. The office_1 room was already viewed in development of the inference adapter; this is expanded evaluation, not a fresh blind test. Not full six-fold S3DIS benchmark.',
        'parameters':'0.04m voxel; 40960 point patches; seed42; coverage threshold0.5; max1000 passes; same fixed grouping thresholds.',
        'seconds_this_run':time.perf_counter()-start,'model':model_meta,'code_sha256':code_hashes,
        'rooms':[{'room':r['room'],'points':r['evaluation']['points'],'accuracy':r['evaluation']['accuracy'],'miou':r['evaluation']['miou_present_union'],'groups':r['group_count'],'unassigned':r['unassigned_group_points'],'inference_seconds':r['prediction']['inference_seconds']} for r in reports]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('BATCH SUMMARY',json.dumps({k:v for k,v in summary.items() if k in ['completed_rooms','mean_room_miou','failures']}),flush=True)
    print('Aggregate accuracy',summary['aggregate']['accuracy'],'mIoU',summary['aggregate']['miou_present_union'],flush=True)
    if failures:raise RuntimeError('Batch contains failures; inspect progress.json')
    return summary

if __name__=='__main__':batch()
