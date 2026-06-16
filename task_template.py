import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18, resnet34, resnet50


NUM_CLASSES = 9
IMAGE_SIZE = 32
MODEL_PATH = "model.pt"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_mean_std(images: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(images).float().div(255.0)
    mean = tensor.mean(dim=(0, 2, 3))
    std = tensor.std(dim=(0, 2, 3)).clamp_min(1e-6)
    return mean, std


class NPZDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mean: torch.Tensor,
        std: torch.Tensor,
        train: bool,
    ) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()
        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)
        self.train = train

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].float().div(255.0)
        label = self.labels[index]

        if self.train:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
            image = F.pad(image, (4, 4, 4, 4))
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            image = image[:, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]

        image = (image - self.mean) / self.std
        return image, label


def build_model() -> tuple[nn.Module, bool]:
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained = True
    except Exception:
        model = resnet18(weights=None)
        pretrained = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model, pretrained


def split_indices(num_items: int, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(num_items)
    val_size = max(1, int(num_items * val_fraction))
    return indices[val_size:], indices[:val_size]


def build_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    pretrained: bool,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    if pretrained:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean, std = compute_mean_std(images)

    train_indices, val_indices = split_indices(len(labels))
    train_dataset = NPZDataset(images, labels, mean, std, train=True)
    val_dataset = NPZDataset(images, labels, mean, std, train=False)

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(val_dataset, val_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, mean, std


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    seed_everything(42)

    data = np.load("train.npz")
    images = data["images"]
    labels = data["labels"]

    device = get_device()
    model, pretrained = build_model()
    model.to(device)

    train_loader, val_loader, mean, std = build_loaders(images, labels, pretrained)

    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        len(labels) / (NUM_CLASSES * np.maximum(class_counts, 1)),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    lr = 3e-4 if pretrained else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(15):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/15 | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        sanity = model(torch.randn(1, 3, 32, 32, device=device))
    assert sanity.shape == (1, 9), sanity.shape

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved best state dict to {MODEL_PATH}")
    print(f"Final checkpoint validation accuracy: {best_val_acc:.4f}")
    print(f"Normalization mean: {mean.tolist()}")
    print(f"Normalization std: {std.tolist()}")


if __name__ == "__main__":
    main()