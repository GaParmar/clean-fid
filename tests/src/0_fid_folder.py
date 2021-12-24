from cleanfid import fid

# bs=128 for default
folder_fake = "tmp/folder_fake"

# check the clean FID
score = fid.compute_fid(folder_fake, dataset_name="horse2zebra",
            dataset_res=256, mode="clean",
            dataset_split="test", batch_size=8, num_workers=24)
print(f"clean-fid score: {score:.4f}")
if abs(score-75.1679)>1e-3:
    raise ValueError(f"clean FID does not match the expected value")

# check the legacy pytorch fid
score = fid.compute_fid(folder_fake, dataset_name="horse2zebra", 
            dataset_res=256, mode="legacy_pytorch",
            dataset_split="test", batch_size=8, num_workers=24)
print(f"legacy-pytorch-fid score: {score:.4f}")
if abs(score-77.2019)>1e-3:
    raise ValueError(f"legacy-pytorch-FID does not match the expected value")

# check the legacy tensorflow fid
score = fid.compute_fid(folder_fake, dataset_name="horse2zebra",
            dataset_res=256, mode="legacy_tensorflow",
            dataset_split="test", batch_size=8, num_workers=24)
print(f"legacy-tensorflow-fid score: {score:.4f}")
if abs(score-76.9118)>1e-3:
    raise ValueError(f"legacy-tensorflow-FID does not match the expected value")

