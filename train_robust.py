import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 9
IMAGE_SIZE = 32
MODEL_PATH = "model.pt"

EPS = 8 / 255          # L-inf budget
ALPHA = 2 / 255        # PGD step size
PGD_STEPS = 3          # PGD-3 steps per training batch
CLEAN_EPOCHS = 5       # warm-up on clean images
ADV_EPOCHS = 95        # PGD-3 adversarial training (~135 min total)
BATCH_SIZE = 128

# Piecewise LR milestones (applied to absolute epoch index)
# Reduces oscillation vs cosine: step down at 40% and 70% of ADV_EPOCHS
LR_MILESTONES = [CLEAN_EPOCHS + int(ADV_EPOCHS * 0.40),
                 CLEAN_EPOCHS + int(ADV_EPOCHS * 0.70)]
LR_GAMMA = 0.1         # multiply LR by this at each milestone


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


class NPZDataset(Dataset):
    """Returns [0, 1] float images — no mean/std normalization."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()
        self.train = train

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].float().div(255.0)
        label = self.labels[idx]
        if self.train:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
            image = F.pad(image, (4, 4, 4, 4))
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            image = image[:, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
        return image, label


def fgsm_rs(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    FGSM with random start (FGSM-RS). Faster than PGD-7; stable when
    paired with random init. Step size = eps (full jump to eps boundary).
    """
    x = images.detach()

    # Random start inside eps ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()

    was_training = model.training
    model.eval()

    delta.requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(x + delta), labels)
    loss.backward()

    with torch.no_grad():
        delta = delta + eps * delta.grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x + delta, 0.0, 1.0) - x

    if was_training:
        model.train()

    return (x + delta.detach()).clamp(0.0, 1.0).detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = EPS,
    alpha: float = 2 / 255,
    steps: int = 7,
) -> torch.Tensor:
    """7-step PGD L-inf attack for evaluation. Images in [0, 1]."""
    x = images.detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()

    was_training = model.training
    model.eval()

    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x + delta), labels)
        loss.backward()
        with torch.no_grad():
            delta = delta + alpha * delta.grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(x + delta, 0.0, 1.0) - x
        delta = delta.detach()

    if was_training:
        model.train()

    return (x + delta).clamp(0.0, 1.0).detach()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    adversarial: bool = True,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        x_train = pgd_attack(model, images, labels, steps=PGD_STEPS) if adversarial else images

        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_train)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    adversarial: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if adversarial:
            images = pgd_attack(model, images, labels)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    data = np.load("train.npz")
    images, labels = data["images"], data["labels"]

    indices = np.random.permutation(len(labels))
    val_size = max(1, int(len(labels) * 0.1))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    train_ds = NPZDataset(images, labels, train=True)
    val_ds = NPZDataset(images, labels, train=False)
    train_loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, val_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print("Pretrained ImageNet weights; BatchNorm will adapt to [0,1] inputs")
    except Exception:
        model = resnet18(weights=None)
        print("Random init")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    total_epochs = CLEAN_EPOCHS + ADV_EPOCHS
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    # Piecewise decay: more stable convergence than cosine for PGD-AT
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    best_state = copy.deepcopy(model.state_dict())
    best_score = 0.0   # composite: 0.5*clean + 0.5*robust (eval every 5 adv epochs)
    best_clean = 0.0

    rob_estimate = 0.0  # running estimate, updated every 5 epochs
    ROB_EVAL_EVERY = 5
    rob_val_loader = DataLoader(Subset(val_ds, val_idx[:512].tolist()), batch_size=128, shuffle=False, num_workers=0)

    for epoch in range(total_epochs):
        is_adv = epoch >= CLEAN_EPOCHS
        phase = f"PGD-{PGD_STEPS}" if is_adv else "clean  "
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, adversarial=is_adv)
        val_loss, val_acc = evaluate(model, val_loader, device, adversarial=False)
        scheduler.step()

        # Update robustness estimate every ROB_EVAL_EVERY adversarial epochs
        if is_adv and (epoch - CLEAN_EPOCHS + 1) % ROB_EVAL_EVERY == 0:
            _, rob_estimate = evaluate(model, rob_val_loader, device, adversarial=True)

        composite = 0.5 * val_acc + 0.5 * rob_estimate
        if composite >= best_score:
            best_score = composite
            best_clean = val_acc
            best_state = copy.deepcopy(model.state_dict())

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{phase}] Epoch {epoch + 1:02d}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_clean={val_acc:.4f} rob~{rob_estimate:.4f} | "
            f"composite={composite:.4f} best={best_score:.4f} lr={lr_now:.5f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        assert model(torch.zeros(1, 3, 32, 32, device=device)).shape == (1, 9)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}  |  best composite score: {best_score:.4f}  (clean={best_clean:.4f})")

    # Final robustness estimate on 512 val samples
    print("Estimating robustness on 512 val samples with PGD-7...")
    val_small = DataLoader(Subset(val_ds, val_idx[:512].tolist()), batch_size=128, shuffle=False, num_workers=0)
    _, rob_acc = evaluate(model, val_small, device, adversarial=True)
    print(f"PGD-7 robust acc (512 samples): {rob_acc:.4f}")
    print(f"Estimated score: {0.5 * best_clean + 0.5 * rob_acc:.4f}")


if __name__ == "__main__":
    main()
