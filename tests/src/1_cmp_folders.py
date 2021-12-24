import torch
from cleanfid import fid

folder_fake = "tmp/folder_fake"
folder_real = "tmp/folder_real"

# check the clean FID
score = fid.compute_fid(folder_real, folder_fake, mode="clean", batch_size=8, 
            num_workers=20, device=torch.device("cpu"), verbose=False)
print(f"\nclean-fid score: {score:.4f}")
if abs(score-75.1679)>1e-3:
    raise ValueError(f"clean FID does not match the expected value")

# check the legacy pytorch fid
score = fid.compute_fid(folder_real, folder_fake, mode="legacy_pytorch", batch_size=8,
            num_workers=20, device=torch.device("cpu"), verbose=False)
print(f"\nlegacy-pytorch-fid score: {score:.4f}")
if abs(score-77.2019)>1e-3:
    raise ValueError(f"legacy pytorch FID does not match the expected value")

# check the legacy tensorflow fid
score = fid.compute_fid(folder_real, folder_fake, mode="legacy_tensorflow", batch_size=8,
            num_workers=20, device=torch.device("cpu"), verbose=False)
print(f"\nlegacy-tensorflow-fid score: {score:.4f}")
if abs(score-76.9118)>1e-3:
    raise ValueError(f"legacy tensorflow FID does not match the expected value")