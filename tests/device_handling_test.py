import pytest
import torch
from cleanfid import fid
from torchvision import datasets, transforms


@pytest.mark.parametrize("mode", ["clean", "legacy_pytorch", "legacy_tensorflow"])
@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_fid_with_non_default_device(mode: str, device: str):
    """Test fid calculation with non-default device"""
    batch_size = 50

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    def gen_images(_) -> torch.Tensor:
        # return cifar10 images
        images = torch.stack([train_dataset[i][0] for i in range(batch_size)])
        # convert to unit8
        images = images.mul(127.5).add_(128).clamp_(0, 255).to("cpu", torch.uint8)
        return images

    fid.compute_fid(
        gen=gen_images,
        dataset_name="cifar10",
        batch_size=batch_size,
        dataset_res=32,
        device=device,
        num_gen=batch_size,
        mode=mode,
    )
