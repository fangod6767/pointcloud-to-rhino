"""Explicit simple-geometry policy; no claim of exact reconstruction."""
import numpy as np
from core import fit_candidate

def generate_candidates(session,thickness=.2,tolerance=.04,boxes=False,progress=None):
    models=[];skipped=[]
    for index,(gid,info) in enumerate(session.info.items()):
        if progress:progress(index,len(session.info))
        if gid<0 or info['class_id']<0:
            skipped.append({'group_id':gid,'reason':'待归组点保留为点云参考'});continue
        points=session.xyz[session.assignment==gid]
        try:
            kind='slab' if info['class_id'] in {0,1,2,5,6,11} else 'box'
            if kind=='box' and not boxes:
                skipped.append({'group_id':gid,'reason':'非平面类别保留点云；未启用近似包围盒'});continue
            try:model=fit_candidate(points,gid,info['name'],kind,thickness,tolerance)
            except ValueError:
                if not boxes or kind=='box':raise
                model=fit_candidate(points,gid,info['name'],'box',thickness,tolerance)
            models.append(model)
        except ValueError as e:skipped.append({'group_id':gid,'reason':str(e)})
    session.models=models
    return {'generated':len(models),'slabs':sum(m['kind']=='slab' for m in models),'boxes':sum(m['kind']=='box' for m in models),'skipped':skipped,'thickness_m':thickness,'plane_tolerance_m':tolerance,'all_geometry_requires_review':True}
