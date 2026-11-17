"""Resumable parallel download; verify complete archive against mirror LFS SHA256."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,json,time,hashlib,os
ROOT=Path(__file__).parent
NAME='Stanford3dDataset_v1.2_Aligned_Version.zip'
URL='https://huggingface.co/datasets/cminst/S3DIS/resolve/main/'+NAME
SIZE=4394701719
SHA256='f16db73310983e6b4df9b3d5290d20315e0cdf56c14f0d795f96d372378fce94'
BLOCK=16*1024*1024

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def download():
    out=ROOT/'data/downloads';out.mkdir(parents=True,exist_ok=True)
    final=out/NAME;part=out/(NAME+'.part');state_path=out/'download_state.json'
    if final.exists():
        if final.stat().st_size==SIZE and file_hash(final)==SHA256:
            print('Already verified:',final,flush=True);return final
        raise ValueError('Existing final archive does not match expected hash; preserve and investigate')
    state=json.loads(state_path.read_text()) if state_path.exists() else {'sha256':SHA256,'done':{}}
    assert state['sha256']==SHA256
    if not part.exists():
        with part.open('wb') as f:f.truncate(SIZE)
        state['done']={}
    assert part.stat().st_size==SIZE
    # Revalidate completed ranges before resuming, rather than trusting the log.
    with part.open('rb') as f:
        for key,digest in list(state['done'].items()):
            start=int(key)*BLOCK;f.seek(start)
            if hashlib.sha256(f.read(min(BLOCK,SIZE-start))).hexdigest()!=digest:del state['done'][key]
    def fetch(i):
        start=i*BLOCK;end=min(start+BLOCK,SIZE)-1
        for attempt in range(6):
            try:
                with requests.get(URL,params={'range_start':start,'range_end':end},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise ValueError('Server did not honor exact range')
                    h=hashlib.sha256();count=0
                    with part.open('r+b') as f:
                        f.seek(start)
                        for b in r.iter_content(1024*1024):
                            count+=len(b)
                            if count>end-start+1:raise ValueError('Oversized range')
                            f.write(b);h.update(b)
                    if count!=end-start+1:raise ValueError('Incomplete range')
                return i,h.hexdigest()
            except (requests.RequestException,ValueError):
                if attempt==5:raise
                time.sleep(min(8,attempt+1))
    count=(SIZE+BLOCK-1)//BLOCK;pending=[i for i in range(count) if str(i) not in state['done']]
    started=time.perf_counter();last=0;initial=len(state['done'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch,i) for i in pending]):
            i,digest=future.result();state['done'][str(i)]=digest
            temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state),encoding='utf8');os.replace(temp,state_path)
            now=time.perf_counter()
            if now-last>5 or len(state['done'])==count:
                done=sum(min(BLOCK,SIZE-int(k)*BLOCK) for k in state['done'])
                rate=(len(state['done'])-initial)*BLOCK/max(now-started,.01)/1e6
                print(f'Download {done/SIZE:.1%} | {done/1e9:.2f}/{SIZE/1e9:.2f} GB | {rate:.1f} MB/s',flush=True);last=now
    print('Verifying complete SHA-256...',flush=True)
    actual=file_hash(part)
    if actual!=SHA256:raise ValueError('Complete archive hash mismatch: '+actual)
    os.replace(part,final)
    provenance={'dataset':'S3DIS v1.2 Aligned Version','download_url':URL,'source_type':'third-party mirror',
        'bytes':SIZE,'sha256':actual,'verification':'matches Hugging Face repository LFS SHA256; not an independently supplied Stanford checksum',
        'license_note':'Mirror license label does not replace original dataset terms.','download_date':'2026-11-11'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print('DOWNLOAD VERIFIED',str(final),flush=True)
    return final

if __name__=='__main__':download()
