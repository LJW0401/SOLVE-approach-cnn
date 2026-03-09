"""
SOLVE - CNN Baseline Approach for Clothing Classification
Fashion-MNIST Dataset | Target Accuracy: 0.95
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import time

# ============ Configuration ============
CONFIG = {
    "approach": "CNN Baseline",
    "iteration": 1,
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ============ Model ============
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============ Data ============
def get_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ============ Training ============
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    print(f"=== SOLVE CNN Baseline - Iteration {CONFIG['iteration']} ===")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")
    print(f"Device: {CONFIG['device']}")

    device = torch.device(CONFIG["device"])
    train_loader, test_loader = get_data_loaders(CONFIG["batch_size"])

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    best_acc = 0
    history = []
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Target: 0.9500")
    print(f"{'PASSED' if best_acc >= 0.95 else 'NOT REACHED'}")
    print(f"Training Time: {elapsed:.1f}s")

    # Save results
    results = {
        "approach": CONFIG["approach"],
        "iteration": CONFIG["iteration"],
        "best_accuracy": best_acc,
        "target": 0.95,
        "passed": best_acc >= 0.95,
        "parameters": param_count,
        "training_time_seconds": elapsed,
        "config": CONFIG,
        "history": history,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/iteration_{CONFIG['iteration']}.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_acc


if __name__ == "__main__":
    main()
