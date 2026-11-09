"""Run with python -m unittest discover -s tests -v."""
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch
from rankwalk import (compute_jaccard_fast, compute_jaccard_sparse, train_gnn,
                      pool_trajectories, build_temporal_graph_grid)
from rankwalk.contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed
from rankwalk.gnn import FeatureAwareRGNN
from rankwalk.build_temporal_graph import _robust_distance_matrix, _final_robust_distance_matrix

HAS_NUMBA = importlib.util.find_spec('numba') is not None

def edges():
    return torch.tensor([[0,1,1,2,2,0],[1,0,2,1,0,2]])


class CorrectnessTests(unittest.TestCase):
    def test_revisits_are_unique_integer_pairs(self):
        e = torch.tensor([[0,1],[1,0]])
        pairs = sample_pos_pairs_start_anchor(torch.zeros(2,2), e, 2, 8, 8)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({(a,b) for a,b,w in pairs}, {(0,1),(1,0)})
        self.assertTrue(all(isinstance(b,int) for a,b,w in pairs))

    def test_isolated_nodes_and_top_one(self):
        for k in [1,5]:
            pairs = sample_pos_pairs_start_anchor(torch.zeros(4,4), edges(), 4, 20, k)
            ids = [(a,b) for a,b,w in pairs]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertTrue(all(a != b and a != 3 for a,b in ids))
            if k == 1:self.assertEqual(pairs, [])

    def test_single_variable_and_unchanged_multivariate_distances(self):
        from scipy.spatial.distance import pdist,squareform
        x = np.array([[0.],[1.],[3.]])
        d = squareform(pdist(x));expected_univariate = d.ravel().argsort().argsort().reshape(d.shape)
        for fn in [_robust_distance_matrix,_final_robust_distance_matrix]:
            np.testing.assert_array_equal(fn(x),expected_univariate)
            with self.assertRaises(ValueError):fn(np.empty((3,0)))
            rng = np.random.default_rng(4);x4 = rng.normal(size=(12,4))
            for ratio in [0.,.1,.7,1.]:
                np.random.seed(31);expected=np.zeros((12,12),np.float32)
                for _ in range(5):
                    cols=np.random.choice(4,max(2,int(np.floor(ratio*4))),replace=False)
                    dd=squareform(pdist(x4[:,cols]));expected += dd.ravel().argsort().argsort().reshape(dd.shape)
                np.random.seed(31)
                np.testing.assert_array_equal(fn(x4,subspace_ratio=ratio),expected/5)

    def test_univariate_graph_builds(self):
        import pandas as pd
        df=pd.DataFrame([(s,t,0,float(s+t),0) for s in range(8) for t in [0.,.5,1.]],columns=['subject','time','outcome','y','cluster'])
        graph,_=build_temporal_graph_grid(df,n_bins=2,k_similarity=2)
        self.assertGreater(graph.number_of_edges(),0)

    def test_jaccard_matches_neighbor_sets(self):
        for seed in range(5):
            rng=np.random.default_rng(seed);ei=torch.from_numpy(rng.integers(0,9,size=(2,35)))
            neighbors=[set() for _ in range(10)]
            for a,b in ei.T.tolist():neighbors[a].add(b);neighbors[b].add(a)
            expected=np.zeros((10,10),np.float32)
            for a in range(10):
                for b in range(10):
                    union=neighbors[a]|neighbors[b]
                    if a!=b and union:expected[a,b]=len(neighbors[a]&neighbors[b])/len(union)
            np.testing.assert_allclose(compute_jaccard_sparse(ei,10).toarray(),expected,atol=1e-7)
            np.testing.assert_allclose(compute_jaccard_fast(ei,10).numpy(),expected,atol=1e-7)

    def test_empty_jaccard_and_edge_validation(self):
        e=torch.empty((2,0),dtype=torch.long)
        self.assertEqual(compute_jaccard_fast(e,0).shape,(0,0))
        self.assertEqual(compute_jaccard_sparse(e,3).nnz,0)
        for invalid in [torch.tensor([[0],[4]]),torch.tensor([[-1],[1]])]:
            with self.assertRaises(ValueError):compute_jaccard_sparse(invalid,3)
        with self.assertRaises(TypeError):compute_jaccard_sparse(edges().float(),3)

    def test_loss_and_gradient_match_original_formula(self):
        pairs=[(0,1,.75),(1,2,.25),(2,3,0.)]
        e=torch.randn(4,5,requires_grad=True);negative=torch.tensor([[2,3],[0,3],[0,1]])
        pos_i=torch.tensor([0,1,2]);pos_j=torch.tensor([1,2,3]);weights=torch.tensor([.75,.25,.05])
        pos=torch.exp((e[pos_i]*e[pos_j]).sum(1)/.5)
        neg=torch.exp((e[pos_i,None,:]*e[negative]).sum(2)/.5).sum(1)
        old=-(weights*torch.log(pos/(pos+neg+1e-8))).mean();old.backward();gradient=e.grad.clone();e.grad.zero_()
        with patch('torch.randint',return_value=negative):new=contrastive_loss_weighted_fixed(e,pairs,neg_multiplier=2)
        new.backward()
        torch.testing.assert_close(new,old,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(e.grad,gradient,atol=1e-6,rtol=1e-6)

    def test_empty_loss_has_zero_gradient(self):
        e=torch.randn(3,4,requires_grad=True);loss=contrastive_loss_weighted_fixed(e,[]);loss.backward()
        self.assertEqual(loss.item(),0.)
        torch.testing.assert_close(e.grad,torch.zeros_like(e))

    def test_large_logits_are_finite(self):
        e=torch.full((3,4),100.,requires_grad=True)
        loss=contrastive_loss_weighted_fixed(e,[(0,1,1.)]);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(e.grad).all())


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_existing_positional_api_and_seed(self):
        x=torch.randn(3,4);ei=edges();j=compute_jaccard_fast(ei,3)
        a=train_gnn(x,ei,None,j,2,.001,5,3,'cpu',seed=4,verbose=False)
        b=train_gnn(x,ei,None,j,epochs=2,walk_length=5,top_k=3,seed=4,verbose=False)
        self.assertIsInstance(a,torch.Tensor);self.assertEqual(a.shape,(3,48));self.assertFalse(a.requires_grad)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a.norm(dim=1),torch.ones(3),atol=1e-5,rtol=1e-5)

    def test_cache_matches_uncached_and_does_not_cross_training_calls(self):
        x=torch.randn(3,4);ei=edges();j=compute_jaccard_fast(ei,3)
        a=train_gnn(x,ei,None,j,epochs=2,seed=4,cache_graph=True,verbose=False)
        b=train_gnn(x,ei,None,j,epochs=2,seed=4,cache_graph=False,verbose=False)
        torch.testing.assert_close(a,b)
        standalone=FeatureAwareRGNN(4)
        self.assertFalse(standalone.conv1.cached)
        # Same standalone model must honor a different graph on its next call.
        standalone.eval();a=standalone(x,ei,None);b=standalone(x,torch.empty((2,0),dtype=torch.long),None)
        self.assertFalse(torch.allclose(a,b))

    def test_edgeless_training(self):
        x=torch.randn(3,4);ei=torch.empty((2,0),dtype=torch.long);j=compute_jaccard_fast(ei,3)
        a=train_gnn(x,ei,None,j,epochs=2,verbose=False)
        self.assertTrue(torch.isfinite(a).all())

    def test_input_errors(self):
        x=torch.randn(3,4);ei=edges();j=compute_jaccard_fast(ei,3)
        for kw in [dict(backend='unknown'),dict(epochs=0),dict(top_k=0),dict(lr=0),dict(backend='numba',device='cuda')]:
            with self.assertRaises(ValueError):train_gnn(x,ei,None,j,verbose=False,**kw)
        with self.assertRaises(TypeError):train_gnn(x,ei,None,compute_jaccard_sparse(ei,3),epochs=1,verbose=False)

    @unittest.skipUnless(HAS_NUMBA,'optional Numba extra not installed')
    def test_numba_training_dense_sparse_and_reproducibility(self):
        x=torch.randn(3,4);ei=edges();j=compute_jaccard_sparse(ei,3)
        a=train_gnn(x,ei,None,j,epochs=3,seed=7,backend='numba',verbose=False)
        b=train_gnn(x,ei,None,torch.tensor(j.toarray()),epochs=3,seed=7,backend='numba',verbose=False)
        torch.testing.assert_close(a,b)
        self.assertEqual(a.shape,(3,48));self.assertTrue(torch.isfinite(a).all())
        e=torch.empty((2,0),dtype=torch.long)
        out=train_gnn(x,e,None,compute_jaccard_sparse(e,3),epochs=1,backend='numba',verbose=False)
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
    def test_cuda_torch_api(self):
        x=torch.randn(3,4);ei=edges();j=compute_jaccard_fast(ei,3,device='cuda')
        out=train_gnn(x,ei,None,j,epochs=2,device='cuda',verbose=False)
        self.assertEqual(out.device.type,'cuda')


@unittest.skipUnless(HAS_NUMBA,'optional Numba extra not installed')
class FastSamplerTests(unittest.TestCase):
    def test_cdf_sampling_respects_directions_and_parallel_edges(self):
        from rankwalk._fast import CPUWalkSampler,_sample_pairs
        # 0 -> 1 twice, 0 -> 2 once; neither neighbor has a return edge.
        ei=torch.tensor([[0,0,0],[1,1,2]]);j=torch.zeros(3,3)
        sampler=CPUWalkSampler(ei,j,3,seed=3);a,m=sampler.adjacency,sampler.similarity
        for uniform,expected in [(.1,1),(.65,1),(.8,2)]:
            u=np.full((3,1),uniform)
            p=_sample_pairs(a.indptr,a.indices,a.data,m.indptr,m.indices,m.data,u,3)
            self.assertEqual(p.shape,(1,3));self.assertEqual(tuple(p[0,:2]),(0,expected))

    def test_weighted_multistep_sampling_matches_dense_cdf_reference(self):
        from rankwalk._fast import CPUWalkSampler,_sample_pairs
        for seed in range(5):
            rng=np.random.default_rng(seed)
            ei=torch.from_numpy(rng.integers(0,6,size=(2,24)))
            similarity=rng.random((6,6));similarity[similarity<.4]=0
            sampler=CPUWalkSampler(ei,torch.tensor(similarity),6)
            uniforms=rng.random((6,12));a,j=sampler.adjacency,sampler.similarity
            actual=_sample_pairs(a.indptr,a.indices,a.data,j.indptr,j.indices,j.data,uniforms,5)
            neighbors=[sorted(ei[1,ei[0]==i].tolist()) for i in range(6)]
            expected=[]
            for start in range(6):
                current=start;visited={start:1}
                for step in range(12):
                    choices=neighbors[current]
                    if not choices:break
                    probabilities=similarity[start,choices]+1e-6
                    cdf=np.cumsum(probabilities/probabilities.sum())
                    current=choices[min(np.searchsorted(cdf,uniforms[start,step],side='right'),len(choices)-1)]
                    if current not in visited:visited[current]=step+2
                chosen=list(visited)[:5];maximum=visited[chosen[-1]]
                expected.extend((start,node,1-visited[node]/(maximum+1e-6)) for node in chosen if node!=start)
            np.testing.assert_allclose(actual,np.asarray(expected).reshape(-1,3),atol=1e-10)

    def test_sparse_torch_input_and_invalid_similarities(self):
        from rankwalk._fast import CPUWalkSampler
        j=compute_jaccard_fast(edges(),3).to_sparse()
        a=CPUWalkSampler(edges(),j,3,seed=1).sample(10,3)
        self.assertEqual(len({tuple(p[:2]) for p in a}),len(a))
        bad=torch.zeros(3,3);bad[0,1]=-1
        with self.assertRaises(ValueError):CPUWalkSampler(edges(),bad,3)
        with self.assertRaises(ValueError):CPUWalkSampler(edges(),torch.zeros(2,2),3)


class ReadoutTests(unittest.TestCase):
    def test_mean_sorted_ids_and_dtype(self):
        e=torch.tensor([[1.,0.],[3.,0.],[0.,1.],[0.,3.]],dtype=torch.float64)
        pooled,ids=pool_trajectories(e,[20,20,5,5],return_subject_ids=True)
        torch.testing.assert_close(pooled,torch.tensor([[0.,2.],[2.,0.]],dtype=torch.float64))
        self.assertEqual(ids.tolist(),[5,20]);self.assertEqual(pooled.dtype,e.dtype)

    def test_ordered_readout_matches_numpy_reference_and_has_gradients(self):
        e=torch.tensor([[1.,2.],[3.,1.],[2.,4.],[5.,3.]],requires_grad=True)
        subjects=np.array([10,10,20,20]);t=np.array([.2,.8,.1,.9]);raw=np.arange(2*16*4).reshape(2,16,4).astype('float32')
        out=pool_trajectories(e,subjects,t,mode='ordered',raw_trajectories=raw)
        a=e.detach().numpy();blocks=[]
        from sklearn.preprocessing import normalize
        mean=np.array([a[subjects==s].mean(0) for s in [10,20]])
        z=np.array([np.concatenate([np.interp(np.linspace(0,1,4),t[subjects==s],a[subjects==s,k]) for k in range(2)]) for s in [10,20]])
        expected=np.concatenate([normalize(mean),normalize(z),normalize(raw.reshape(2,-1))],1)
        np.testing.assert_allclose(out.detach().numpy(),expected,atol=1e-6)
        self.assertEqual(out.shape,(2,74));out.sum().backward();self.assertTrue(torch.isfinite(e.grad).all())

    def test_single_nodes_duplicates_and_physical_time_grid(self):
        e=torch.tensor([[1.,2.],[4.,3.],[6.,5.]])
        a=pool_trajectories(e,[1,1,2],[0.,0.,1.],mode='ordered')
        self.assertTrue(torch.isfinite(a).all())
        expected=torch.nn.functional.normalize(torch.tensor([[4.,4.,4.,4.,3.,3.,3.,3.]]),dim=1)
        torch.testing.assert_close(a[0:1,2:],expected)
        out=pool_trajectories(e,[1,1,2],[10.,10.,20.],mode='ordered',time_grid=[10.,15.,20.])
        self.assertEqual(out.shape,(2,8))

    def test_zero_half_precision_blocks_are_finite(self):
        emb=torch.zeros(2,4,dtype=torch.float16)
        result=pool_trajectories(emb,[0,1],[0.,1.],mode='ordered',raw_trajectories=torch.zeros(2,16,1))
        self.assertEqual(result.dtype,torch.float16)
        self.assertTrue(torch.isfinite(result).all())

    def test_readout_errors(self):
        e=torch.randn(2,3)
        for kw in [dict(mode='invalid'),dict(mode='ordered'),dict(mode='ordered',times=[3.,4.]),dict(raw_trajectories=torch.zeros(2,16,1)),dict(mode='ordered',times=[0.,1.],time_grid=[1.,0.])]:
            with self.assertRaises(ValueError):pool_trajectories(e,[0,1],**kw)

if __name__=='__main__':unittest.main()
