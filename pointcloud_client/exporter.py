"""Native editable Rhino geometry exported without requiring Rhino installation."""
from pathlib import Path
import json,os,re
import numpy as np
import rhino3dm as r
from core import CLASS_ZH,group_color,model_vertices

def group_display_name(gid,info):
    """Use the current reviewed class/name, including for older saved projects."""
    cid=info['class_id']
    label=CLASS_ZH[cid] if cid>=0 else ('待归组' if gid<0 else '混合待审阅')
    number=f'G{gid+1:03d}' if gid>=0 else '未分配'
    custom=str(info.get('name','')).strip()
    custom=re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', custom)
    suffix='_'+custom if custom and custom not in (number,label,'待归组') else ''
    return f'{label}_{number}{suffix}'

def native_geometry(model):
    if model['kind']=='mesh':
        mesh=r.Mesh();mesh.Vertices.UseDoublePrecisionVertices=True
        for p in model_vertices(model):mesh.Vertices.Add(*map(float,p))
        for f in model['faces']:mesh.Faces.AddFace(*map(int,f))
        for c in model['colors']:mesh.VertexColors.Add(*map(int,c))
        mesh.Normals.ComputeNormals();mesh.Compact()
        return mesh
    lo=model['low'];hi=model['high'];basis=np.array(model['basis']);center=model['center']
    if model['kind']=='surface':
        plane=r.Plane(r.Point3d(*center),r.Vector3d(*basis[:,0]),r.Vector3d(*basis[:,1]))
        return r.Brep.CreateFromSurface(r.PlaneSurface(plane,r.Interval(lo[0],hi[0]),r.Interval(lo[1],hi[1])))
    brep=r.Brep.CreateFromBoundingBox(r.BoundingBox(r.Point3d(*lo),r.Point3d(*hi)))
    transform=r.Transform.Identity()
    for i in range(3):
        for j in range(3):setattr(transform,f'M{i}{j}',float(basis[i,j]))
        setattr(transform,f'M{i}3',float(center[i]))
    if not brep.Transform(transform):raise ValueError('几何坐标转换失败。')
    return brep

def export_3dm(session,path,include_clouds=True,progress=None):
    path=Path(path);file=r.File3dm();file.Settings.ModelUnitSystem=r.UnitSystem.Meters;file.Settings.ModelAbsoluteTolerance=.0001
    def layer(name,color,parent=None,visible=True):
        item=r.Layer();item.Name=name;item.Color=tuple(color)+(255,);item.Visible=visible
        if parent is not None:item.ParentLayerId=file.Layers[parent].Id
        return file.Layers.Add(item)
    cloud_root=layer('01_点云分组',(100,144,170),visible=not bool(session.models))
    model_root=layer('02_建模结果',(37,175,157))
    model_layers={}
    cloud_points=0;clouds=0;model_count=0;solids=0;mesh_count=0;mesh_faces=0
    if include_clouds:
        for k,(gid,info) in enumerate(sorted(session.info.items())):
            cid=info['class_id'];color=tuple((group_color(gid)*255).astype(int));name=group_display_name(gid,info)
            li=layer(name,color,cloud_root)
            attr=r.ObjectAttributes();attr.Name=name;attr.LayerIndex=li;attr.ObjectColor=color+(255,);attr.ColorSource=r.ObjectColorSource.ColorFromObject
            attr.SetUserString('CandidateGroup',str(gid));attr.SetUserString('ReviewedClass',CLASS_ZH[cid] if cid>=0 else 'mixed')
            attr.SetUserString('Source',session.metadata.get('source_file',''));attr.SetUserString('GeometryRole','Segmented reference point cloud; not a solid')
            cloud=r.PointCloud();idx=np.flatnonzero(session.assignment==gid)
            for p in session.xyz[idx]:cloud.Add(r.Point3d(*p))
            file.Objects.AddPointCloud(cloud,attr);cloud_points+=len(idx);clouds+=1
            if progress:progress(int(65*(k+1)/len(session.info)),f'导出点云组 {k+1}/{len(session.info)}')
    for i,model in enumerate(session.models):
        geometry=native_geometry(model)
        if not geometry.IsValid:raise ValueError('无效几何：'+model['name'])
        is_mesh=model['kind']=='mesh'
        if model['kind'] not in ['surface','mesh'] and not geometry.IsSolid:raise ValueError('体块没有闭合：'+model['name'])
        gid=model['group_id'];info=session.info[gid];cid=info['class_id']
        name=group_display_name(gid,info)
        if gid not in model_layers:
            model_layers[gid]=layer(name,tuple((group_color(gid)*255).astype(int)),model_root)
        attr=r.ObjectAttributes();attr.Name=name+'_'+model['kind'];attr.LayerIndex=model_layers[gid]
        attr.SetUserString('ReviewedClass',CLASS_ZH[cid] if cid>=0 else 'mixed')
        attr.SetUserString('CandidateGroup',str(model['group_id']));attr.SetUserString('ModelParameters',json.dumps({k:v for k,v in model.items() if k not in ['vertices','faces','colors']},ensure_ascii=False))
        if is_mesh:
            attr.SetUserString('GeometryRole','Measured-surface mesh; open boundaries allowed; not a CAD solid')
            file.Objects.AddMesh(geometry,attr);mesh_count+=1;mesh_faces+=len(model['faces'])
        else:file.Objects.AddBrep(geometry,attr);solids+=int(geometry.IsSolid)
        model_count+=1
        if progress:progress(65+int(25*(i+1)/max(1,len(session.models))),'导出可编辑几何')
    file.Strings['PointCloudDesk']=json.dumps({'version':1,'source':session.metadata,'coordinate_units':'meters','cloud_points':cloud_points,'model_count':model_count},ensure_ascii=False)
    temp=path.with_name(path.stem+'.writing.3dm')
    if not file.Write(str(temp),8):raise OSError('3DM 写入失败。')
    check=r.File3dm.Read(str(temp))
    if check is None or len(check.Objects)!=clouds+model_count:raise OSError('3DM 回读校验失败。')
    if not all(o.Geometry.IsValid for o in check.Objects):raise OSError('3DM 包含无效对象。')
    if sum(len(o.Geometry.Faces) for o in check.Objects if isinstance(o.Geometry,r.Mesh))!=mesh_faces:raise OSError('3DM 网格面数回读不一致。')
    os.replace(temp,path)
    return {'path':str(path),'cloud_objects':clouds,'cloud_points':cloud_points,'models':model_count,'closed_solids':solids,'meshes':mesh_count,'mesh_faces':mesh_faces,'units':'meters','readback_valid':True}
