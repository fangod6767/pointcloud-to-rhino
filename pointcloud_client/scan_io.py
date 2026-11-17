"""Raw scan readers. Units/axes are explicit; input labels are never read."""
from pathlib import Path
import struct
import numpy as np
from plyfile import PlyData

MAX_POINTS=10_000_000

def lzf_decode(data,expected):
    out=bytearray();i=0
    while i<len(data):
        control=data[i];i+=1
        if control<32:
            count=control+1
            if i+count>len(data):raise ValueError('PCD LZF 字面数据截断')
            out.extend(data[i:i+count]);i+=count
        else:
            count=control>>5;offset=(control&31)<<8
            if count==7:
                if i>=len(data):raise ValueError('PCD LZF 长度截断')
                count+=data[i];i+=1
            if i>=len(data):raise ValueError('PCD LZF 偏移截断')
            offset+=data[i]+1;i+=1;count+=2
            if offset>len(out):raise ValueError('PCD LZF 偏移无效')
            for _ in range(count):out.append(out[-offset])
        if len(out)>expected:raise ValueError('PCD 解压长度超出声明')
    if len(out)!=expected:raise ValueError('PCD 解压长度不一致')
    return bytes(out)

def read_pcd(path):
    with Path(path).open('rb') as f:
        header={}
        for _ in range(100):
            line=f.readline(65536).decode('ascii').strip()
            if not line or line.startswith('#'):continue
            key,*value=line.split();header[key.upper()]=value
            if key.upper()=='DATA':break
        else:raise ValueError('PCD 文件头无效')
        names=header.get('FIELDS',header.get('FIELD',[]));sizes=list(map(int,header['SIZE']))
        types=header['TYPE'];counts=list(map(int,header.get('COUNT',['1']*len(names))))
        n=int(header['POINTS'][0]) if 'POINTS' in header else int(header['WIDTH'][0])*int(header.get('HEIGHT',['1'])[0])
        if not 1<=n<=MAX_POINTS:raise ValueError('请按房间拆分点云，每次最多处理 1000 万点')
        if not len(names)==len(sizes)==len(types)==len(counts) or len(set(names))!=len(names):raise ValueError('PCD 字段声明无效')
        dtype=[]
        for name,size,kind,count in zip(names,sizes,types,counts):
            if kind not in ['F','I','U'] or size not in [1,2,4,8] or not 1<=count<=64:raise ValueError('PCD 字段类型不支持')
            fmt=np.dtype('<'+{'F':'f','I':'i','U':'u'}[kind]+str(size))
            dtype.append((name,fmt) if count==1 else (name,fmt,(count,)))
        dtype=np.dtype(dtype);mode=header['DATA'][0].lower()
        if mode=='ascii':
            raw=np.loadtxt(f,ndmin=2)
            if raw.shape!=(n,sum(counts)):raise ValueError('PCD 点数或列数与文件头不一致')
            data=np.empty(n,dtype);column=0
            for name,count in zip(names,counts):
                data[name]=raw[:,column] if count==1 else raw[:,column:column+count];column+=count
        elif mode=='binary':
            raw=f.read(n*dtype.itemsize)
            if len(raw)!=n*dtype.itemsize:raise ValueError('PCD 二进制数据截断')
            data=np.frombuffer(raw,dtype,count=n)
        elif mode=='binary_compressed':
            size_header=f.read(8)
            if len(size_header)!=8:raise ValueError('PCD 压缩头截断')
            compressed,uncompressed=struct.unpack('<II',size_header)
            if uncompressed!=n*dtype.itemsize or compressed>uncompressed*2+1024:raise ValueError('PCD 压缩长度异常')
            raw=lzf_decode(f.read(compressed),uncompressed);data=np.empty(n,dtype);offset=0
            for name in names:
                field=dtype.fields[name][0];size=n*field.itemsize
                data[name]=np.frombuffer(raw[offset:offset+size],field,count=n);offset+=size
        else:raise ValueError('不支持的 PCD DATA 类型')
    return fields_to_points(data)

def fields_to_points(data):
    names=set(data.dtype.names or [])
    if not {'x','y','z'}.issubset(names):raise ValueError('点云缺少 x/y/z 字段')
    xyz=np.column_stack([data[a] for a in ['x','y','z']]).astype(np.float64)
    rgb=None
    for channels in [('red','green','blue'),('r','g','b')]:
        if set(channels).issubset(names):rgb=np.column_stack([data[a] for a in channels]);break
    if rgb is None and ('rgb' in names or 'rgba' in names):
        packed=data['rgb' if 'rgb' in names else 'rgba']
        packed=np.ascontiguousarray(packed,dtype='<f4').view('<u4') if packed.dtype.kind=='f' else packed.astype(np.uint32)
        rgb=np.column_stack([(packed>>16)&255,(packed>>8)&255,packed&255])
    return xyz,rgb

def read_scan(path,scale=1.,up_axis='z',neutral=False):
    path=Path(path);ext=path.suffix.lower()
    if ext=='.pcd':xyz,rgb=read_pcd(path)
    elif ext=='.ply':
        ply=PlyData.read(path);vertex=ply['vertex']
        if vertex.count>MAX_POINTS:raise ValueError('每次最多处理 1000 万点，请按房间拆分')
        xyz,rgb=fields_to_points(vertex.data)
    elif ext=='.npz':
        with np.load(path,allow_pickle=False) as d:
            xyz=np.array(d['xyz'],dtype=np.float64);rgb=np.array(d['rgb']) if 'rgb' in d else None
    elif ext in ['.xyz','.xyzrgb','.txt','.pts']:
        with path.open('r',encoding='utf8') as f:
            first=f.readline();declared=None
            if ext=='.pts' and len(first.split())==1:declared=int(first)
            else:f.seek(0)
            data=np.loadtxt(f,ndmin=2)
        if data.shape[1] not in [3,6,7]:raise ValueError('文本格式需要 XYZ、XYZ RGB 或 XYZ intensity RGB 列')
        if declared is not None and len(data)!=declared:raise ValueError('PTS 声明点数与实际不一致')
        xyz=data[:,:3];rgb=data[:,-3:] if data.shape[1]>=6 else None
        if rgb is not None and rgb.max()<=1:rgb=rgb*255.
    else:raise ValueError('支持 PCD / PLY / XYZ / XYZRGB / TXT / PTS / NPZ；E57、LAS 请先转换')
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not 32<=len(xyz)<=MAX_POINTS:raise ValueError('点云需包含 32 至 1000 万个点，请按室内场景处理')
    original_count=len(xyz);valid=np.isfinite(xyz).all(1)
    if rgb is not None:
        if rgb.shape!=xyz.shape:raise ValueError('RGB 与点坐标数量不一致')
        valid&=np.isfinite(rgb).all(1)
        if np.any((rgb[valid]<0)|(rgb[valid]>255)):raise ValueError('RGB 需在 0–255 范围内')
    xyz=xyz[valid];rgb=rgb[valid] if rgb is not None else None
    if len(xyz)<32:raise ValueError('清理无效坐标后不足 32 个点')
    missing=rgb is None
    if missing and not neutral:raise ValueError('该扫描没有 RGB。当前室内模型需要颜色；可返回勾选“无颜色实验模式”，但其识别效果尚未验证。')
    if missing:rgb=np.full(xyz.shape,127.5,np.float32)
    if not np.isfinite(scale) or scale<=0 or up_axis not in ['x','y','z']:raise ValueError('单位或竖直轴无效')
    rotation={'z':np.eye(3),'y':np.array([[1,0,0],[0,0,-1],[0,1,0]]),'x':np.array([[0,0,-1],[0,1,0],[1,0,0]])}[up_axis]
    canonical=xyz@rotation.T*scale
    # Stabilize float32 inference for survey/global coordinates. Retain source
    # position in exported geometry, but bring local scan's floor near zero.
    origin=np.r_[np.median(canonical[:,:2],axis=0),canonical[:,2].min()]
    network=(canonical-origin).astype(np.float32)
    return xyz,network,rgb.astype(np.float32),dict(source=str(path.resolve()),scale_to_meters=scale,up_axis=up_axis,network_rotation=rotation.tolist(),inference_origin_m=origin.tolist(),missing_rgb_fallback=missing,input_points=original_count,removed_nonfinite=original_count-len(xyz),point_count=len(xyz))
