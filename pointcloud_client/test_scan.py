from pathlib import Path
import tempfile,unittest,struct,json
import numpy as np
from scan_io import read_scan,read_pcd,lzf_decode
from core import Session
from auto_geometry import generate_candidates

class ScanTests(unittest.TestCase):
    def test_ascii_binary_and_compressed_pcd(self):
        a=np.zeros(40,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]);a['x']=np.arange(40);a['y']=2;a['z']=3;a['rgb']=(23<<16)|(134<<8)|201
        common='VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH 40\nHEIGHT 1\nPOINTS 40\nDATA '
        with tempfile.TemporaryDirectory() as d:
            for mode in ['ascii','binary','binary_compressed']:
                path=Path(d)/(mode+'.pcd')
                if mode=='ascii':payload='\n'.join(' '.join(str(v) for v in row) for row in a).encode()
                elif mode=='binary':payload=a.tobytes()
                else:
                    raw=b''.join(a[k].tobytes() for k in a.dtype.names)
                    compressed=b''.join(bytes([len(raw[i:i+32])-1])+raw[i:i+32] for i in range(0,len(raw),32))
                    payload=struct.pack('<II',len(compressed),len(raw))+compressed
                path.write_bytes((common+mode+'\n').encode()+payload)
                xyz,rgb=read_pcd(path);np.testing.assert_array_equal(xyz[:,0],np.arange(40));np.testing.assert_array_equal(rgb,np.tile([23,134,201],(40,1)))
    def test_lzf_backreference_and_bad_size(self):
        self.assertEqual(lzf_decode(bytes([2])+b'abc'+bytes([128,2]),9),b'abcabcabc')
        with self.assertRaises(ValueError):lzf_decode(bytes([32,4]),3)
    def test_units_up_axis_cleaning_and_local_origin(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.npz';xyz=np.tile([1e9,2000,3000],(40,1));xyz[0,0]=np.nan
            np.savez(p,xyz=xyz,rgb=np.full(xyz.shape,100),semantic=np.full(40,99))
            original,network,rgb,meta=read_scan(p,.001,'y')
            self.assertEqual(len(original),39);self.assertEqual(meta['removed_nonfinite'],1)
            canonical=original@np.array(meta['network_rotation']).T*.001
            np.testing.assert_allclose(network+meta['inference_origin_m'],canonical)
            self.assertLess(abs(network).max(),1e-6)
    def test_uncolored_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.xyz';np.savetxt(p,np.random.default_rng(5).normal(size=(40,3)))
            with self.assertRaises(ValueError):read_scan(p)
            self.assertTrue(read_scan(p,neutral=True)[3]['missing_rgb_fallback'])
    def test_auto_policy_preserves_nonplanar_points(self):
        a,b=np.meshgrid(np.linspace(0,2,10),np.linspace(0,3,10));plane=np.c_[a.ravel(),b.ravel(),a.ravel()*0]
        rng=np.random.default_rng(42);cloud=rng.uniform(4,5,(100,3));xyz=np.r_[plane,cloud]
        s=Session(xyz,np.full(xyz.shape,100),np.r_[np.ones(100),np.full(100,8)],np.ones(200),np.r_[np.zeros(100),np.ones(100)])
        result=generate_candidates(s);self.assertEqual(result['slabs'],1);self.assertEqual(result['boxes'],0);self.assertEqual(len(s.xyz),200)
        result=generate_candidates(s,boxes=True);self.assertEqual(result['boxes'],1);self.assertEqual(len(s.xyz),200)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScanTests))
    out=Path(__file__).parent/'qa';out.mkdir(exist_ok=True)
    (out/'scan_reader_tests.json').write_text(json.dumps({'success':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)},indent=2))
    raise SystemExit(not result.wasSuccessful())
