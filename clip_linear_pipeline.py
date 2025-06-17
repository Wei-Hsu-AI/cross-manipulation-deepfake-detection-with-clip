import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from dataset import build_dataloaders
from utils import save_prediction_results, seed_everything, evaluate_metrics


class CLIPLinearClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        for param in self.clip.parameters():
            param.requires_grad = False  # Freeze all CLIP weights
        self.classifier = nn.Linear(self.clip.config.projection_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            feats = self.clip.get_image_features(pixel_values=images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.classifier(feats)


def train(model, loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


def evaluate(model, loader, device):
    model.eval()
    preds, scores, labels, paths = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            label = batch["label"].item()
            path = batch["path"]
            logits = model(images)
            prob = logits.softmax(dim=-1)[0]
            pred = prob.argmax().item()
            preds.append(pred)
            scores.append(prob[1].item())
            labels.append(label)
            paths.append(path)
    return preds, scores, labels, paths


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str,
                        default="results/baseline_clip_linear")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_loader, val_loader = build_dataloaders(args.data_root, processor)
    model = CLIPLinearClassifier().to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    for e in range(args.epochs):
        print(f"Epoch {e+1}")
        train(model, train_loader, optimizer, scheduler, device)

    preds, scores, labels, paths = evaluate(model, val_loader, device)
    metrics = evaluate_metrics(scores, preds, labels)
    print(
        f"AUC: {metrics['AUC']:.3f}, F1: {metrics['F1']:.3f}, Accuracy: {metrics['Accuracy']:.3f}, EER: {metrics['EER']:.3f}")

    save_prediction_results(
        paths=paths,
        scores=scores,
        labels=labels,
        preds=preds,
        results_path=args.results_dir
    )


if __name__ == "__main__":
    main()
