"""Meaningful editing, geometry and real-data native-file integration checks."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
import rhino3dm as r
from core import Session,fit_candidate
from exporter import export_3dm,native_geometry
ROOT=Path(__file__).parent
SAMPLE=ROOT.parent/'results/area3_batch/Area_3_office_2/segmentation.npz'

def sample_session():
    a,b=np.meshgrid(np.linspace(0,2,20),np.linspace(0,3,25));p=np.c_[a.ravel(),b.ravel(),a.ravel()*0]
    return Session(p,np.full(p.shape,180,np.uint8),np.ones(len(p),np.uint8),np.ones(len(p)),(p[:,0]>1).astype(np.int32))

class ClientTests(unittest.TestCase):
    def test_merge_split_history_preserves_points_and_predictions(self):
        s=sample_session();xyz=s.xyz.copy();pred=s.prediction.copy();original=s.assignment.copy()
        merged=s.merge([0,1],1);self.assertEqual(len(s.info),1)
        ids=s.split(merged,0,1.);self.assertEqual(len(s.info),2);self.assertEqual(len(s.assignment),len(xyz))
        self.assertTrue(s.undo());self.assertEqual(len(s.info),1)
        self.assertTrue(s.undo());np.testing.assert_array_equal(s.assignment,original)
        self.assertTrue(s.undo(True));self.assertEqual(len(s.info),1)
        np.testing.assert_array_equal(s.xyz,xyz);np.testing.assert_array_equal(s.prediction,pred)
    def test_invalid_split_is_atomic(self):
        s=sample_session();before=s.assignment.copy()
        with self.assertRaises(ValueError):s.split(0,0,20)
        np.testing.assert_array_equal(s.assignment,before);self.assertFalse(s.dirty)
    def test_group_changes_invalidate_old_geometry_and_undo_restores(self):
        s=sample_session();s.models=[fit_candidate(s.xyz[s.assignment==0],0,'a')]
        s.merge([0,1],1);self.assertFalse(s.models);s.undo();self.assertEqual(len(s.models),1)
    def test_project_roundtrip(self):
        s=sample_session();s.rename(0,'地板 A',1);s.models=[fit_candidate(s.xyz[s.assignment==0],0,'地板 A')]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'工程.pcproj';s.save(p);restored=Session.load(p)
            np.testing.assert_array_equal(restored.xyz,s.xyz);np.testing.assert_array_equal(restored.assignment,s.assignment)
            self.assertEqual(restored.info,s.info);self.assertEqual(restored.models,s.models)
    def test_nonplanar_data_rejected_for_slab(self):
        p=np.random.default_rng(5).uniform(-1,1,(3000,3))
        with self.assertRaises(ValueError):fit_candidate(p,0,'不规则物体',tolerance=.04)
        g=native_geometry(fit_candidate(p,0,'包围盒',kind='box'));self.assertTrue(g.IsValid and g.IsSolid)
    def test_rotated_plane_geometry_and_dimensions(self):
        s=sample_session();p=s.xyz.copy();angle=.7;R=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        p=p@R.T+[10,20,30]
        for kind in ['surface','slab']:
            m=fit_candidate(p,0,'斜板',kind,thickness=.25);g=native_geometry(m)
            self.assertTrue(g.IsValid);self.assertEqual(g.IsSolid,kind=='slab');self.assertLess(m['p95_m'],1e-10)
            if kind=='slab':self.assertAlmostEqual(m['high'][2]-m['low'][2],.25)
    def test_native_export_has_all_points_and_closed_solid(self):
        s=sample_session();s.models=[fit_candidate(s.xyz,0,'板体',thickness=.2)]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'test.3dm';summary=export_3dm(s,p);file=r.File3dm.Read(str(p))
            self.assertEqual(summary['cloud_points'],len(s.xyz));self.assertEqual(summary['closed_solids'],1)
            self.assertEqual(file.Settings.ModelUnitSystem,r.UnitSystem.Meters)
            self.assertEqual(sum(len(o.Geometry) for o in file.Objects if isinstance(o.Geometry,r.PointCloud)),len(s.xyz))
    def test_real_room_export_and_source_preservation(self):
        s=Session.load(SAMPLE)
        with np.load(SAMPLE) as d:np.testing.assert_array_equal(s.xyz,d['xyz'])
        floor=max((gid for gid,info in s.info.items() if info['class_id']==1),key=lambda gid:np.count_nonzero(s.assignment==gid))
        s.models=[fit_candidate(s.xyz[s.assignment==floor],floor,'示例地板',thickness=.2)]
        out=ROOT/'qa';out.mkdir(exist_ok=True)
        summary=export_3dm(s,out/'office2_demo.3dm');s.save(out/'office2_demo.pcproj')
        self.assertEqual(summary['cloud_points'],739209);self.assertEqual(summary['closed_solids'],1)
        (out/'export_check.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'qa').mkdir(exist_ok=True)
    (ROOT/'qa/core_test_results.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful()},indent=2))
    raise SystemExit(not result.wasSuccessful())
