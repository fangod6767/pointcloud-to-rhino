from pathlib import Path
import unittest,tempfile,json
import numpy as np
from scipy.spatial import cKDTree
import rhino3dm as r
from core import Session,model_vertices,fit_candidate
from mesh_reconstruction import reconstruct_mesh
from exporter import export_3dm,native_geometry

def perforated_plane():
    x,y=np.meshgrid(np.arange(0,1.001,.02),np.arange(0,1.001,.02))
    p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    return p[~((p[:,0]>.32)&(p[:,0]<.68)&(p[:,1]>.32)&(p[:,1]<.68))]

class MeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points=perforated_plane();cls.colors=np.tile([40,160,220],(len(cls.points),1)).astype(np.uint8)
        cls.model=reconstruct_mesh(cls.points,cls.colors,0,'带洞表面',.02,.08)
    def session(self):
        n=len(self.points);return Session(self.points,self.colors,np.ones(n),np.ones(n),np.zeros(n),models=[self.model])
    def test_hole_not_bridged_and_edge_limit(self):
        m=self.model;v=model_vertices(m);t=v[m['faces']];centroids=t.mean(1)
        self.assertFalse(np.any((centroids[:,0]>.35)&(centroids[:,0]<.65)&(centroids[:,1]>.35)&(centroids[:,1]<.65)))
        self.assertLessEqual(np.linalg.norm(t-np.roll(t,1,axis=1),axis=2).max(),.080001)
        self.assertGreater(m['boundary_edges'],0);self.assertGreater(len(m['faces']),1000)
    def test_original_point_vertices_and_real_curvature(self):
        p=self.points.copy();p[:,2]=.08*np.sin(p[:,0]*8)
        m=reconstruct_mesh(p,self.colors,0,'波形',.02,.08)
        v=model_vertices(m);self.assertLess(cKDTree(p).query(v)[0].max(),1e-10)
        self.assertGreater(np.ptp(v[:,2]),.12)
    def test_project_roundtrip_and_undo(self):
        s=self.session()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mesh.pcproj';s.save(path);loaded=Session.load(path)
            for key in ['vertices','faces','colors']:np.testing.assert_array_equal(s.models[0][key],loaded.models[0][key])
            loaded.clear_models();self.assertFalse(loaded.models);loaded.undo();self.assertEqual(len(loaded.models),1)
            np.testing.assert_array_equal(loaded.models[0]['faces'],s.models[0]['faces'])
    def test_native_mesh_is_open_colored_and_survives_3dm(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mesh.3dm';report=export_3dm(self.session(),path);doc=r.File3dm.Read(str(path))
            meshes=[o.Geometry for o in doc.Objects if isinstance(o.Geometry,r.Mesh)]
            self.assertEqual(len(meshes),1);mesh=meshes[0]
            self.assertTrue(mesh.IsValid);self.assertFalse(mesh.IsClosed)
            self.assertEqual(len(mesh.Faces),len(self.model['faces']));self.assertEqual(len(mesh.VertexColors),len(self.model['vertices']))
            self.assertEqual(report['closed_solids'],0);self.assertEqual(report['meshes'],1)
    def test_large_coordinate_origin_and_invalid_parameters(self):
        shifted=dict(self.model,center=(np.array(self.model['center'])+[1e6,2e6,3e6]).tolist())
        mesh=native_geometry(shifted);bounds=mesh.GetBoundingBox()
        self.assertAlmostEqual(bounds.Min.X,1e6,places=5);self.assertAlmostEqual(bounds.Max.X,1e6+1,places=5)
        with self.assertRaises(ValueError):reconstruct_mesh(self.points,self.colors,0,'bad',.02,.01)

    def test_export_names_follow_reviewed_class_for_mesh_and_brep(self):
        s=self.session()
        # A saved geometry name may be stale; current reviewed group info wins.
        s.info[0].update(name='入口:地板',class_id=2)
        s.models.append(fit_candidate(s.xyz,0,'旧名字',kind='slab'))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'names.3dm';export_3dm(s,p);doc=r.File3dm.Read(str(p))
            objects=list(doc.Objects)
            self.assertEqual(len(objects),3)
            for obj in objects:
                name=doc.Layers[obj.Attributes.LayerIndex].Name
                self.assertEqual(name,'墙_G001_入口_地板')
                self.assertTrue(obj.Attributes.Name.startswith(name))
                self.assertEqual(obj.Attributes.GetUserString('ReviewedClass'),'墙')
            geometry=[o for o in objects if not isinstance(o.Geometry,r.PointCloud)]
            self.assertEqual(geometry[0].Attributes.LayerIndex,geometry[1].Attributes.LayerIndex)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MeshTests))
    (Path(__file__).parent/'qa/mesh_unit_results.json').write_text(json.dumps({'success':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)},indent=2))
    raise SystemExit(not result.wasSuccessful())
