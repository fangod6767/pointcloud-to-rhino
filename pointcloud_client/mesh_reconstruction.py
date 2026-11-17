"""Conservative per-group Ball Pivoting: original-point vertices, bounded gaps."""
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

def reconstruct_mesh(points,rgb,gid,name,spacing=.02,max_edge=.08):
    points=np.asarray(points,dtype=np.float64);rgb=np.asarray(rgb,dtype=np.uint8)
    if len(points)<32:raise ValueError('少于 32 个点，保留为点云')
    if not np.isfinite([spacing,max_edge]).all() or spacing<=0 or max_edge<2*spacing:raise ValueError('最大连边长度需至少为采样间距的 2 倍')
    center=points.mean(0);local=points-center
    original=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local));tree=cKDTree(local)
    actual=float(spacing)
    for _ in range(12):
        sampled=original.voxel_down_sample(actual)
        if len(sampled.points)<=200000:break
        actual*=1.4
    if len(sampled.points)>200000:raise ValueError('单组过大，请增大网格采样间距或拆分分组')
    if max_edge<2*actual:raise ValueError('该组为控制内存增大了采样间距，请同时增大最大连边长度或拆组')
    # Select measured points nearest voxel centroids; never project onto a plane.
    indices=np.unique(tree.query(np.asarray(sampled.points),workers=-1)[1])
    p=local[indices]
    if len(p)<32:raise ValueError('采样后点过少，请减小采样间距')
    _,singular,v=np.linalg.svd(p-p.mean(0),full_matrices=False)
    if singular[1]<1e-8:raise ValueError('点集近似直线，不能重建表面')
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=actual*4,max_nn=40))
    normal_policy='local tangent-plane consistency'
    if singular[2]<singular[1]*.06:
        normal=v[-1];normal*=1 if normal[np.argmax(abs(normal))]>=0 else -1
        cloud.orient_normals_to_align_with_direction(normal)
        normal_policy='align estimated normals to dominant plane; vertices remain unchanged'
    else:
        try:cloud.orient_normals_consistent_tangent_plane(min(30,len(p)-1))
        except RuntimeError as e:raise ValueError('法线方向估计失败，请拆分该组后重试') from e
    radii=sorted(set([min(actual*1.5,max_edge/2),min(actual*2.5,max_edge/2)]))
    mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud,o3d.utility.DoubleVector(radii))
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    if not len(faces):raise ValueError('未形成可靠三角面，请调整间距/连边或保留点云')
    triangles=vertices[faces]
    longest=np.max(np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2),axis=1)
    support_limit=actual*1.25
    # Check every triangle centroid and edge midpoint against this group's raw points.
    probes=np.concatenate([triangles.mean(1)[:,None,:],(triangles+np.roll(triangles,1,axis=1))/2],axis=1)
    support=tree.query(probes.reshape(-1,3),workers=-1)[0].reshape(-1,4).max(1)
    keep=(longest<=max_edge+1e-10)&(support<=support_limit)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles)==0:raise ValueError('所有三角面均超出点云支持范围，未强行补面')
    mesh.compute_vertex_normals()
    vertices=np.asarray(mesh.vertices).copy();faces=np.asarray(mesh.triangles,dtype=np.int32).copy()
    # BPA keeps input sample vertices. Recover actual scan RGB at each vertex.
    nearest=tree.query(vertices,workers=-1)[1];colors=rgb[nearest]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    # Bidirectional evidence: raw points -> reconstructed surface, surface -> raw cloud.
    sampled_raw=local[np.linspace(0,len(local)-1,min(5000,len(local)),dtype=int)]
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distance=scene.compute_distance(o3d.core.Tensor(sampled_raw.astype(np.float32))).numpy()
    support_distance=tree.query(vertices[faces].mean(1),workers=-1)[0]
    return dict(kind='mesh',group_id=int(gid),name=name,center=center.tolist(),vertices=vertices,faces=faces,colors=colors,
                points=len(points),sampling_m=actual,requested_spacing_m=float(spacing),max_edge_m=float(max_edge),
                normal_policy=normal_policy,radii_m=radii,vertices_count=len(vertices),triangles_count=len(faces),
                boundary_edges=int(np.sum(counts==1)),closed=bool(np.all(counts==2)),
                surface_to_cloud_p95_m=float(np.quantile(support_distance,.95)),
                cloud_to_surface_p95_m=float(np.quantile(distance,.95)),
                covered_fraction=float(np.mean(distance<=actual*2)),
                assumption='Ball Pivoting from measured points. No hole filling or watertight-solid guarantee. Parameters may bridge gaps smaller than the support/edge thresholds.')

def generate_meshes(session,spacing=.02,max_edge=.08,gids=None,progress=None):
    ids=list(session.info) if gids is None else list(dict.fromkeys(map(int,gids)))
    if not set(ids).issubset(session.info):raise ValueError('选中分组已经变化，请重新选择')
    models=[];skipped=[]
    for i,gid in enumerate(ids):
        if progress:progress(i,len(ids),session.info[gid]['name'])
        idx=np.flatnonzero(session.assignment==gid)
        try:models.append(reconstruct_mesh(session.xyz[idx],session.rgb[idx],gid,session.info[gid]['name'],spacing,max_edge))
        except ValueError as e:skipped.append({'group_id':gid,'reason':str(e)})
    # Retain prior geometry for failed groups; successful groups are replaced.
    session.invalidate({m['group_id'] for m in models});session.models.extend(models)
    return dict(method='Ball Pivoting + local point support and edge filtering',generated=len(models),meshes=len(models),
                vertices=sum(m['vertices_count'] for m in models),triangles=sum(m['triangles_count'] for m in models),
                spacing_m=spacing,max_edge_m=max_edge,skipped=skipped,all_geometry_requires_review=True,
                groups=[{k:v for k,v in m.items() if k not in ['vertices','faces','colors']} for m in models])
