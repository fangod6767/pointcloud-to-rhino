"""Safely extract the verified archive and convert all rooms, recording bad rows."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import zipfile,shutil,json,time,re,io,hashlib,os
import numpy as np
ROOT=Path(__file__).parent
CLASSES=['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

def save_npz_fast(path,**arrays):
    temporary=path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,array in arrays.items():
            with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,array,allow_pickle=False)
    os.replace(temporary,path)

def extract():
    archive=ROOT/'data/downloads/Stanford3dDataset_v1.2_Aligned_Version.zip'
    provenance=json.loads((archive.parent/'provenance.json').read_text())
    assert archive.stat().st_size==provenance['bytes']
    destination=ROOT/'data/full_raw';destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'extraction_verified.json'
    if marker.exists():return destination
    total=0;last=0;start=time.perf_counter()
    with zipfile.ZipFile(archive) as z:
        for i,info in enumerate(z.infolist()):
            target=(destination/info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe ZIP path: '+info.filename)
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            # ZipExtFile checks the CRC when the member has been fully consumed.
            assert target.stat().st_size==info.file_size
            total+=info.file_size
            if time.perf_counter()-last>10:
                print(f'Extracted {i+1}/{len(z.infolist())} entries, {total/1e9:.2f} GB; CRC verified',flush=True);last=time.perf_counter()
    marker.write_text(json.dumps({'archive_sha256':provenance['sha256'],'uncompressed_bytes':total,'all_member_crc_verified':True,'seconds':time.perf_counter()-start},indent=2))
    return destination

def load_annotation(path):
    rejected=[]
    try:values=np.loadtxt(path,dtype=np.float32,ndmin=2,encoding='utf8')
    except (ValueError,UnicodeError):
        rows=[]
        with path.open(encoding='utf8',errors='replace') as f:
            for line_no,line in enumerate(f,1):
                try:
                    fields=line.split()
                    if len(fields)!=6:raise ValueError('expected 6 fields')
                    rows.append([float(x) for x in fields])
                except ValueError:rejected.append({'line':line_no,'text':line.strip()[:160],'reason':'malformed row'})
        values=np.asarray(rows,dtype=np.float32).reshape(-1,6)
    if values.shape[1]!=6:raise ValueError('Unexpected annotation columns: '+str(path))
    valid=np.isfinite(values).all(1)&(values[:,3:]>=0).all(1)&(values[:,3:]<=255).all(1)
    for row in np.flatnonzero(~valid):rejected.append({'parsed_row':int(row),'reason':'nonfinite XYZ/RGB or RGB outside [0,255]'})
    return values[valid],rejected

def convert_room(folder,out):
    area=folder.parent.parent.name;room=folder.parent.name;identifier=area+'_'+room
    target=out/(identifier+'.npz');record_path=out/(identifier+'.json')
    if target.exists() and record_path.exists():return json.loads(record_path.read_text(encoding='utf8'))
    xyz=[];rgb=[];semantic=[];instances=[];records=[]
    for instance,path in enumerate(sorted(folder.glob('*.txt'))):
        values,rejected=load_annotation(path);category=path.stem.rsplit('_',1)[0]
        mapped=category if category in CLASSES else 'clutter'
        xyz.append(values[:,:3]);rgb.append(np.rint(values[:,3:]).astype(np.uint8))
        semantic.append(np.full(len(values),CLASSES.index(mapped),np.uint8));instances.append(np.full(len(values),instance,np.int32))
        records.append({'file':path.name,'original_class':category,'mapped_class':mapped,'points':len(values),'rejected_rows':rejected})
    arrays={'xyz':np.concatenate(xyz),'rgb':np.concatenate(rgb),'semantic':np.concatenate(semantic),'instance':np.concatenate(instances)}
    assert len(arrays['xyz'])>0 and all(len(a)==len(arrays['xyz']) for a in arrays.values())
    save_npz_fast(target,**arrays)
    record={'room':area+'/'+room,'file':str(target.relative_to(ROOT)),'points':len(arrays['xyz']),'instances':len(records),
        'class_counts':np.bincount(arrays['semantic'],minlength=13).tolist(),'rejected_rows':sum(len(r['rejected_rows']) for r in records),
        'unknown_class_files':[r['file'] for r in records if r['original_class']!=r['mapped_class']],
        'annotations':records,'policy':'Original files retained; malformed/nonfinite/out-of-range rows excluded and explicitly logged. Unknown object classes map to clutter, following Open3D-ML S3DIS.'}
    record_path.write_text(json.dumps(record,indent=2),encoding='utf8');return record

def prepare():
    start=time.perf_counter();raw=extract();out=ROOT/'data/full_processed';out.mkdir(parents=True,exist_ok=True)
    folders=sorted(p for p in raw.rglob('Annotations') if p.is_dir() and re.fullmatch('Area_[1-6]',p.parent.parent.name))
    assert len(folders)==272
    reports=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(convert_room,p,out) for p in folders]):
            record=future.result();reports.append(record)
            print(f"Converted {len(reports)}/{len(folders)} {record['room']} {record['points']:,} points, rejected {record['rejected_rows']}",flush=True)
    summary={'dataset':'S3DIS v1.2 Aligned Version','room_count':len(reports),'total_points':sum(r['points'] for r in reports),
        'total_rejected_rows':sum(r['rejected_rows'] for r in reports),'classes':CLASSES,'seconds':time.perf_counter()-start,
        'rooms':[{'room':r['room'],'file':r['file'],'points':r['points'],'rejected_rows':r['rejected_rows'],'unknown_class_files':r['unknown_class_files']} for r in sorted(reports,key=lambda r:r['room'])]}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('ALL ROOMS CONVERTED',summary['room_count'],'points',summary['total_points'],'rejected',summary['total_rejected_rows'],flush=True)

if __name__=='__main__':prepare()
