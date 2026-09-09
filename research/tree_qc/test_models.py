import unittest
import torch
from fp32_policy import configure_fp32
FP32_POLICY = configure_fp32()
from models import MODELS, farthest_point_sample, radius_group

class PointOpsTests(unittest.TestCase):
    def test_fps_coverage_and_batch_independence(self):
        xyz = torch.tensor([[[0.,0,0],[1.,0,0],[3.,0,0],[10.,0,0]]])
        both = torch.cat([xyz, xyz * 10 + 40], 0)
        idx = farthest_point_sample(both, 4)
        self.assertEqual(idx[0].unique().numel(), 4)
        self.assertTrue(torch.equal(idx[:1], farthest_point_sample(xyz, 4)))
        self.assertTrue(torch.equal(idx[:1], idx[1:]))
        self.assertEqual(set(idx[0,:2].tolist()), {0,3})

    def test_radius_members_and_padding(self):
        xyz = torch.tensor([[[0.,0,0],[1.,0,0],[3.,0,0],[10.,0,0]]])
        idx = radius_group(xyz, xyz[:,:1], 1.1, 4)
        self.assertEqual(set(idx.flatten().tolist()), {0,1})
        self.assertEqual((idx == 0).sum().item(), 3)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA gate environment')
    def test_sizes_gradients_and_model_batch_independence(self):
        for factory in MODELS.values():
            model = factory().cuda().eval()
            for count in [2048,4096]:
                x = torch.randn(2,count,4,device='cuda')
                x[...,3] = (x[...,3]>0).float()
                logits = model(x)
                self.assertEqual(tuple(logits.shape),(2,2))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.allclose(logits[:1],model(x[:1]),atol=1e-5,rtol=1e-4))
                logits.square().sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                model.zero_grad(set_to_none=True)

if __name__ == '__main__':
    unittest.main()
