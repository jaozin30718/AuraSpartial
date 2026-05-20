import torch
import torch.nn.functional as F

def check_shapes(T_m, F_m):
    mask_real = torch.randn(1, 2, T_m, F_m)
    mix_stft = torch.randn(1, 257, 501)
    
    T_mask, F_mask = mask_real.shape[2], mask_real.shape[3]
    F_stft, T_stft = mix_stft.shape[1], mix_stft.shape[2]

    if F_stft > F_mask:
        mix_stft = mix_stft[:, :F_mask, :]
    elif F_stft < F_mask:
        mix_stft = F.pad(mix_stft, (0, 0, 0, F_mask - F_stft))

    if T_stft > T_mask:
        mix_stft = mix_stft[:, :, :T_mask]
    elif T_stft < T_mask:
        mix_stft = F.pad(mix_stft, (0, T_mask - T_stft))
        
    X = mix_stft.unsqueeze(1)
    M_r = mask_real.transpose(2, 3)
    print(f"Input mask: {T_m}x{F_m} -> M_r: {M_r.shape}, X: {X.shape}")

check_shapes(501, 128)
check_shapes(128, 501)
