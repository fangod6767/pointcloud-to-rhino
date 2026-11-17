"""Retrieve one annotated S3DIS room using byte ranges; retain provenance."""
import io, zipfile, requests, json, hashlib, sys, argparse, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT=Path(__file__).parent
URL='https://huggingface.co/datasets/cminst/S3DIS/resolve/main/Stanford3dDataset_v1.2_Aligned_Version.zip'
SIZE=4394701719
CLASSES=['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

class RemoteZip(io.RawIOBase):
    def __init__(self):self.pos=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        if n==0:return b''
        if n>32*1024*1024:raise ValueError('Unexpected large range')
        start=self.pos;end=min(start+n,SIZE)-1
        for attempt in range(4):
            try:
                with requests.get(URL,params={'range_start':start,'range_end':end},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=30) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Range request not honored; aborting full download')
                    b=r.content
                break
            except requests.RequestException:
                if attempt==3:raise
                time.sleep(1+attempt)
        if len(b)!=end-start+1:raise ValueError('Short range read')
        self.pos+=len(b);self.transferred+=len(b);return b

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--room',default='Area_1/office_1')
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'Area_[1-6]/[A-Za-z]+_\d+',args.room):raise ValueError('Invalid room identifier')
    area,room_name=args.room.split('/')
    ROOT.mkdir(exist_ok=True)
    remote=RemoteZip()
    with zipfile.ZipFile(remote) as z:
        entries=[x for x in z.infolist() if '/'+args.room+'/Annotations/' in x.filename and x.filename.endswith('.txt')]
        if not entries:raise ValueError('Room not found')
        print('Room:',args.room,'annotation files:',len(entries),'compressed bytes:',sum(x.compress_size for x in entries),flush=True)
        meta=[{'name':x.filename,'header_offset':x.header_offset,'compressed_size':x.compress_size,'file_size':x.file_size,'CRC':x.CRC} for x in entries]
    (ROOT/f'remote_{area}_{room_name}_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    if args.inspect:return
    import struct,zlib,binascii
    def fetch(info):
        dest=ROOT/'data'/'raw'/area/room_name/'Annotations'/Path(info.filename).name
        data=dest.read_bytes() if dest.exists() else b''
        if len(data)!=info.file_size or binascii.crc32(data)&0xffffffff!=info.CRC:
            f=RemoteZip();f.seek(info.header_offset);h=f.read(30)
            vals=struct.unpack('<4s5H3I2H',h)
            name_len,extra_len=vals[-2:]
            f.seek(info.header_offset+30+name_len+extra_len)
            raw=f.read(info.compress_size)
            data=zlib.decompress(raw,-15) if info.compress_type==8 else raw
            if len(data)!=info.file_size or binascii.crc32(data)&0xffffffff!=info.CRC:raise ValueError('CRC mismatch')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        values=np.loadtxt(io.BytesIO(data),dtype=np.float32,ndmin=2)
        category=dest.stem.rsplit('_',1)[0]
        if category not in CLASSES:raise ValueError('Unexpected class '+category)
        return values,CLASSES.index(category),{'filename':dest.name,'points':len(values),'sha256':hashlib.sha256(data).hexdigest()}
    parts=[];labels=[];instances=[];files=[]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i,(values,label,metadata) in enumerate(executor.map(fetch,entries)):
            parts.append(values);labels.append(np.full(len(values),label,dtype=np.uint8));instances.append(np.full(len(values),i,dtype=np.int16));files.append(metadata)
            print('Loaded',metadata['filename'],metadata['points'],flush=True)
    values=np.concatenate(parts);labels=np.concatenate(labels);instances=np.concatenate(instances)
    assert values.shape[1]==6 and np.isfinite(values).all()
    out=ROOT/'data'/'processed';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'{area}_{room_name}.npz',xyz=values[:,:3],rgb=values[:,3:].astype(np.uint8),semantic=labels,instance=instances)
    provenance={'dataset':'S3DIS v1.2 aligned','room':args.room,'original_paper':'https://openaccess.thecvf.com/content_cvpr_2016/html/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.html','download_mirror':URL,'mirror_note':'Third-party mirror; mirror license tag is not taken as original dataset license. Preserve original dataset attribution and check original terms before redistribution.','classes':CLASSES,'points':len(values),'counts':{CLASSES[int(k)]:int(v) for k,v in zip(*np.unique(labels,return_counts=True))},'files':files,'use':'Area 3 is held out according to the official Open3D-ML RandLA-Net S3DIS configuration; this subset is not the full benchmark.' if area=='Area_3' else 'Development sample only; not an independent test set.'}
    (ROOT/'data'/f'{area}_{room_name}_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    if args.room=='Area_1/office_1':(ROOT/'data'/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print('READY',len(values),'points',provenance['counts'],flush=True)

if __name__=='__main__':main()
