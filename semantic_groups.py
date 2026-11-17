"""Geometry-based candidate groups INSIDE predicted semantic classes.

These are editable candidates, not a trained instance-segmentation model.
All distances and bounds in this module are in network coordinates (meters).
"""
import numpy as np
from sklearn.cluster import DBSCAN

NAMES=['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

def deterministic_plane(points,distance,rng,iterations=500):
    """Fixed trial order and first-wins tie handling, independent of RANSAC threads."""
    origin=points.mean(0);p=np.asarray(points,dtype=np.float64)-origin
    triples=np.array([rng.choice(len(p),3,replace=False) for _ in range(iterations)])
    samples=p[triples]
    normals=np.cross(samples[:,1]-samples[:,0],samples[:,2]-samples[:,0])
    length=np.linalg.norm(normals,axis=1);valid=length>1e-10
    normals=normals[valid]/length[valid,None]
    if not len(normals):return np.zeros(4),np.array([],dtype=np.int64)
    offsets=-np.einsum('ij,ij->i',normals,samples[valid,0])
    best_count=-1;best=None
    for first in range(0,len(normals),32):
        # Explicit arithmetic fixes reduction order and avoids BLAS thread variability.
        n=normals[first:first+32];d=offsets[first:first+32]
        residual=np.abs(p[:,0,None]*n[None,:,0]+p[:,1,None]*n[None,:,1]+p[:,2,None]*n[None,:,2]+d)
        counts=np.count_nonzero(residual<=distance,axis=0);local=int(counts.argmax())
        if counts[local]>best_count:best_count=int(counts[local]);best=first+local
    normal=normals[best];offset=offsets[best]
    mask=np.abs(p[:,0]*normal[0]+p[:,1]*normal[1]+p[:,2]*normal[2]+offset)<=distance
    # Refit the winning consensus without using any class truth or instance labels.
    q=p[mask];center=q.mean(0);v=q-center
    _,vectors=np.linalg.eigh(v.T@v);normal=vectors[:,0]
    if normal[np.argmax(abs(normal))]<0:normal=-normal
    offset=-normal@center
    inliers=np.flatnonzero(abs(p[:,0]*normal[0]+p[:,1]*normal[1]+p[:,2]*normal[2]+offset)<=distance)
    return np.r_[normal,offset-normal@origin],inliers

def split_semantic_groups(xyz,semantic,confidence,eps=.14,min_points=35,plane_distance=.035,seed=42):
    xyz=np.asarray(xyz);semantic=np.asarray(semantic);confidence=np.asarray(confidence)
    groups=np.full(len(xyz),-1,np.int32);manifest=[]
    rng=np.random.default_rng(seed)
    def add_components(indices,kind,plane=None):
        if len(indices)<min_points:return
        labels=DBSCAN(eps=eps,min_samples=3,n_jobs=-1).fit_predict(xyz[indices])
        for label in sorted(set(labels)-{-1}):
            idx=indices[labels==label]
            if len(idx)<min_points:continue
            gid=len(manifest);groups[idx]=gid;cid=int(semantic[idx[0]])
            record={'id':gid,'class_id':cid,'class_name':NAMES[cid],'method':kind,'coarse_points':len(idx),
                'bounds_m':[xyz[idx].min(0).tolist(),xyz[idx].max(0).tolist()],
                'mean_confidence':float(confidence[idx].mean()),'low_confidence_fraction':float((confidence[idx]<.6).mean()),
                'review_required':True}
            if plane is not None:record['plane_m']=np.asarray(plane).tolist()
            manifest.append(record)
    # Architectural sheet-like classes: distinguish intersecting walls by plane.
    planar={0,1,2,5,6,11}
    for cid in range(len(NAMES)):
        remaining=np.flatnonzero(semantic==cid)
        if cid in planar:
            for _ in range(20):
                if len(remaining)<max(80,min_points):break
                plane,inliers=deterministic_plane(xyz[remaining],plane_distance,rng)
                if len(inliers)<max(80,.05*len(remaining)):break
                inliers=np.asarray(inliers,dtype=np.int64)
                add_components(remaining[inliers],'RANSAC plane + DBSCAN',plane)
                keep=np.ones(len(remaining),bool);keep[inliers]=False;remaining=remaining[keep]
        add_components(remaining,'DBSCAN within predicted class')
    return groups,manifest
