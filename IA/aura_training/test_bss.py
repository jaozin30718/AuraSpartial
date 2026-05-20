import torch
from losses.bss_loss import BSSLoss

# Mock data
B = 1
N = 2
T = 501
F = 128
S = 64000

# masks is [B, N*2, T, F]
masks = torch.randn(B, N*2, T, F)
# stereo is [B, 2, S]
stereo = torch.randn(B, 2, S)
# references is [B, N, S]
references = torch.randn(B, N, S)

loss_fn = BSSLoss()
try:
    loss, sdr = loss_fn(masks, stereo, references)
    print("Success!", loss)
except Exception as e:
    import traceback
    traceback.print_exc()
