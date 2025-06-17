import os
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
from utils import save_prediction_results, seed_everything, evaluate_metrics, load_lora_adapter
from dataset import build_dataloaders


def wrap_clip_with_lora(model_name, lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]):
    clip = CLIPModel.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules
    )
    clip.vision_model = get_peft_model(clip.vision_model, lora_config)

    for name, param in clip.named_parameters():
        if "vision_model" not in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in clip.parameters())
    trainable_params = sum(p.numel()
                           for p in clip.parameters() if p.requires_grad)
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    return clip


def encode_prompts(prompts, clip, processor, device="cuda"):
    with torch.no_grad():
        text_inputs = processor(
            text=prompts, return_tensors="pt", padding=True).to(device)
        text_feats = clip.get_text_features(**text_inputs)
        return text_feats / text_feats.norm(dim=-1, keepdim=True)


def train(clip, epochs, text_feats, train_loader, optimizer, scheduler, device):
    for epoch in range(epochs):
        clip.train()
        running_loss = 0
        for batch in tqdm(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            vision_feats = clip.get_image_features(pixel_values=images)
            vision_feats = vision_feats / \
                vision_feats.norm(dim=-1, keepdim=True)
            logits = vision_feats @ text_feats.T * 10
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


def evaluate_model(clip, text_feats, val_loader, device):
    clip.eval()
    preds, scores, labels, paths = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch["image"].to(device)
            label = batch["label"].item()
            path = batch["path"]

            # 取出圖片特徵並正規化
            vision_feat = clip.get_image_features(pixel_values=image)
            vision_feat = vision_feat / vision_feat.norm(dim=-1, keepdim=True)

            # 與文字 prompt 做 cosine 相似度後 softmax
            logit = (vision_feat @ text_feats.T).softmax(dim=-1)
            pred = logit.argmax(dim=-1).item()
            prob_fake = logit[0, 1].item()  # 取出 class=1 (fake) 的機率

            preds.append(pred)
            scores.append(prob_fake)
            labels.append(label)
            paths.append(path)

    return preds, scores, labels, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt0", type=str, default="AI-Generated face")
    parser.add_argument("--prompt1", type=str, default="Authentic face")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train", help="執行模式：train 或 eval")

    args = parser.parse_args()

    # 自動命名資料夾
    suffix = f"{args.prompt0.replace(' ', '_')}_vs_{args.prompt1.replace(' ', '_')}"
    if args.save_dir is None:
        args.save_dir = os.path.join("lora_checkpoints", suffix)
    if args.results_dir is None:
        args.results_dir = os.path.join("results", suffix)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [args.prompt0, args.prompt1]
    processor = CLIPProcessor.from_pretrained(args.model_name)

    if args.mode == "train":
        clip = wrap_clip_with_lora(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        clip.to(device)

        train_loader, val_loader = build_dataloaders(args.data_root, processor)

        text_feats = encode_prompts(prompts, clip, processor, device)

        optimizer = torch.optim.AdamW(clip.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        train(clip, args.epochs, text_feats,
              train_loader, optimizer, scheduler, device)
        os.makedirs(args.save_dir, exist_ok=True)
        clip.vision_model.save_pretrained(args.save_dir)
        print(f"✅ LoRA adapter saved to: {args.save_dir}")

    clip = load_lora_adapter(args.model_name, args.save_dir)
    clip.to(device)
    preds, scores, labels, paths = evaluate_model(
        clip, text_feats, val_loader, device)

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
