"""Use IEEE FP32 for validation/evaluation; explicit AMP autocast remains float16."""
import torch

def configure_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "amp_policy": "existing explicit float16 autocast unchanged"}
