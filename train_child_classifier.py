"""Train a child vs adult binary classifier on PA-100K surveillance images."""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm


DATA_DIR = '/home/cahlen/dev/gguf-workbench/pa100k/PA-100K'
IMG_DIR = os.path.join(DATA_DIR, 'data')
OUT_DIR = '/home/cahlen/dev/gguf-workbench/child-classifier'


class PA100KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df['AgeLess18'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['Image'])).convert('RGB')
        label = int(row['AgeLess18'])
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    device = 'cuda'
    epochs = 20
    bs = 48
    lr = 2e-4
    os.makedirs(OUT_DIR, exist_ok=True)

    # Augmentation for training - simulate surveillance conditions
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = PA100KDataset(f'{DATA_DIR}/train.csv', IMG_DIR, train_tf)
    val_ds = PA100KDataset(f'{DATA_DIR}/val.csv', IMG_DIR, val_tf)

    # Weighted sampler to handle class imbalance (5% children vs 95% adults)
    labels = train_ds.labels
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train: {len(train_ds)} ({class_counts[1]} children / {class_counts[0]} adults)")
    print(f"Val: {len(val_ds)}")

    # ConvNeXt-Tiny — strong modern backbone, 28M params
    model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=2)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: EfficientNet-B2 ({total_params:.1f}M params)")

    # Moderate child weighting — enough to handle imbalance, not so much it calls everything a child
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))

    best_score = 0  # balanced accuracy: (child_recall + adult_recall) / 2

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        correct, total, loss_sum = 0, 0, 0

        for i, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            loss_sum += loss.item() * len(labels)

        # Validate
        model.train(False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn)

        print(f"Epoch {ep+1}/{epochs} ({time.time()-t0:.0f}s) "
              f"loss={loss_sum/total:.3f} train_acc={correct/total:.1%}")
        print(f"  Child: precision={precision:.1%} recall={recall:.1%} F1={f1:.3f}")
        print(f"  Overall: acc={acc:.1%} (TP={tp} FP={fp} TN={tn} FN={fn})")

        adult_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + adult_recall) / 2
        score = balanced_acc

        if score > best_score:
            best_score = score
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': 'convnext_tiny.fb_in22k_ft_in1k',
                'class_names': ['adult', 'child'],
                'input_size': 224,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }, f'{OUT_DIR}/child_classifier.pt')
            print(f"  -> Saved (balanced_acc={score:.3f} recall={recall:.1%} adult_recall={adult_recall:.1%})")

    print(f"\nDone. Best balanced_acc: {best_score:.3f} -> {OUT_DIR}/child_classifier.pt")


if __name__ == '__main__':
    main()
