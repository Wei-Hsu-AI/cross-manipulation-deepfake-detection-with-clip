import os
import torch
from PIL import Image
from transformers import CLIPModel
from peft import PeftModel
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_lora_adapter(model_name: str, adapter_dir: str):
    """
    載入原始 CLIP 模型並套用 LoRA adapter 權重。

    參數：
    - model_name: 原始 CLIP 模型的名稱或路徑（如 "openai/clip-vit-base-patch32"）
    - adapter_dir: 儲存好的 LoRA adapter 權重資料夾

    回傳：
    - 包含 LoRA 權重的模型（CLIPModel），可直接推論使用
    """
    # 載入 CLIP 模型
    clip = CLIPModel.from_pretrained(model_name)

    # 套用 vision_model 的 LoRA 權重
    peft_model = PeftModel.from_pretrained(clip.vision_model, adapter_dir)
    clip.vision_model = peft_model

    return clip


def save_prediction_results(
    paths,
    scores,
    labels,
    preds,
    results_path
):
    """
    儲存 frame-level 與 video-level 的預測結果為 JSON 檔。

    參數：
    - paths: frame 對應的路徑（每個 path 是 list，來自 batch["path"]）
    - scores: 每個 frame 的 fake 分數（class=1 的機率）
    - labels: 每個 frame 的真實標籤
    - preds: 每個 frame 的預測類別
    - results_path: 結果輸出資料夾（將自動建立 frame_results.json 和 video_results.json）
    """
    os.makedirs(results_path, exist_ok=True)
    frame_output_path = os.path.join(results_path, "frame_results.json")
    video_output_path = os.path.join(results_path, "video_results.json")

    # ---------- Frame-level ----------
    frame_results = []
    for p, s, l, pred in zip(paths, scores, labels, preds):
        frame_results.append({
            "path": p[0],  # p 是 list，取第一張圖路徑
            "score": s,
            "label": l,
            "pred": pred
        })

    with open(frame_output_path, "w") as f:
        json.dump(frame_results, f, indent=2)
    print(f"✅ Saved frame-level results to {frame_output_path}")

    # ---------- Video-level ----------
    video_scores = defaultdict(list)
    video_labels = {}

    for p, s, l in zip(paths, scores, labels):
        vid = os.path.basename(os.path.dirname(p[0]))
        video_scores[vid].append(s)
        video_labels[vid] = l

    video_results = []
    for vid in video_scores:
        avg_score = sum(video_scores[vid]) / len(video_scores[vid])
        video_results.append({
            "video_id": vid,
            "avg_score": avg_score,
            "label": video_labels[vid]
        })

    with open(video_output_path, "w") as f:
        json.dump(video_results, f, indent=2)
    print(f"✅ Saved video-level results to {video_output_path}")


def load_prediction_results(results_path):
    """
    從指定資料夾讀取 frame_results.json 並還原出 paths, scores, labels, preds。

    參數：
    - results_path: 結果資料夾路徑，需包含 frame_results.json 檔案

    回傳：
    - paths, scores, labels, preds（皆為 list）
    """
    frame_output_path = os.path.join(results_path, "frame_results.json")

    if not os.path.exists(frame_output_path):
        raise FileNotFoundError(f"{frame_output_path} 不存在")

    with open(frame_output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paths = [[item["path"]] for item in data]
    scores = [item["score"] for item in data]
    labels = [item["label"] for item in data]
    preds = [item["pred"] for item in data]

    return paths, scores, labels, preds


def evaluate_metrics(scores, preds, labels):
    """
    計算 AUC、F1、Accuracy、EER 分類指標。

    參數：
    - scores: 預測為 fake (class=1) 的機率分數
    - preds: 預測類別（0 或 1）
    - labels: 真實標籤

    回傳：
    - 指標 dict，包含 AUC、F1、Accuracy、EER
    """
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2

    return {
        "AUC": auc,
        "F1": f1,
        "Accuracy": acc,
        "EER": eer,
        "FPR": fpr,
        "TPR": tpr
    }


def plot_roc_curve(fpr, tpr, auc, title="ROC Curve"):
    """
    繪製 ROC 曲線。

    參數：
    - fpr: False Positive Rates
    - tpr: True Positive Rates
    - auc: Area Under Curve 數值
    - title: 圖表標題
    """
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_video_level_accuracy_from_dir(results_path: str, threshold: float = 0.5) -> float:
    """
    從 results 資料夾中讀取 video_results.json 並計算 video-level accuracy。

    Args:
        results_path (str): 包含 video_results.json 的資料夾路徑
        threshold (float): 分類為 fake 的分數閾值（預設為 0.5）

    Returns:
        float: video-level 分類準確率（0~1）
    """
    video_results_path = os.path.join(results_path, "video_results.json")

    if not os.path.isfile(video_results_path):
        raise FileNotFoundError(
            f"找不到 video_results.json: {video_results_path}")

    with open(video_results_path, "r", encoding="utf-8") as f:
        video_results = json.load(f)

    correct = 0
    total = len(video_results)
    for result in video_results:
        pred = 1 if result["avg_score"] >= threshold else 0
        if pred == result["label"]:
            correct += 1

    return correct / total if total > 0 else 0.0


def get_misclassified_neuraltextures_samples_from_file(
    results_path: str,
    num_samples: int = 3
):
    """
    從 frame_results.json 中讀取 NeuralTextures 中的誤判樣本。

    參數：
    - results_path: 儲存結果的資料夾路徑（需包含 frame_results.json）
    - num_samples: 隨機選擇的影片數量（每部影片取一張錯誤的 frame）

    回傳：
    - selected_samples: list of dict，包含 path, pred, label, score, video_id
    """
    frame_result_file = os.path.join(results_path, "frame_results.json")
    with open(frame_result_file, "r") as f:
        frame_results = json.load(f)

    # 篩選出來自 NeuralTextures 且被誤判的 frame
    misclassified = [
        {
            "path": item["path"],
            "pred": item["pred"],
            "label": item["label"],
            "score": item["score"],
            "video_id": os.path.basename(os.path.dirname(item["path"]))
        }
        for item in frame_results
        if "NeuralTextures" in item["path"] and item["pred"] != item["label"]
    ]

    # 按影片分組後隨機挑 num_samples 部影片
    grouped = {}
    for item in misclassified:
        grouped.setdefault(item["video_id"], []).append(item)
    selected_videos = random.sample(
        list(grouped.keys()), k=min(num_samples, len(grouped)))

    # 每部影片中取第一張錯誤 frame
    selected_samples = [grouped[vid][0] for vid in selected_videos]
    return selected_samples


def visualize_misclassified_samples(samples, root_dir=None):
    """
    視覺化 get_misclassified_neuraltextures_samples 回傳的誤判圖片樣本。

    參數：
    - samples: list of dicts，每個 dict 包含 path, video_id, label, pred, score
    - root_dir: 若 path 為相對路徑時用來補完整路徑
    """
    for sample in samples:
        # 組成圖片路徑
        if root_dir and not os.path.isabs(sample["path"]) and not sample["path"].startswith(root_dir):
            img_path = os.path.join(root_dir, sample["path"])
        else:
            img_path = sample["path"]

        image = Image.open(img_path).convert("RGB")

        # 顯示圖片
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"[{sample['video_id']}]\nTrue: {sample['label']} | Pred: {sample['pred']} | Score: {sample['score']:.2f}")
        plt.show()


def analyze_misclassified_with_prompts(samples, clip, processor, prompts, root_dir=None, device="cuda"):
    for sample in samples:
        if root_dir and not os.path.isabs(sample["path"]) and not sample["path"].startswith(root_dir):
            img_path = os.path.join(root_dir, sample["path"])
        else:
            img_path = sample["path"]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # 預處理圖片
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_outputs = clip.vision_model(
                pixel_values=inputs["pixel_values"], output_hidden_states=True)
            patch_feats = vision_outputs.hidden_states[-1][0, 1:]
            patch_feats = clip.visual_projection(patch_feats)  # 變成 (49, 512)

        patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)

        # 預處理所有 prompts
        with torch.no_grad():
            text_inputs = processor(
                text=prompts, return_tensors="pt", padding=True).to(device)
            text_feats = clip.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # 每個 prompt 畫 heatmap
        for i, prompt in enumerate(prompts):
            sim = patch_feats @ text_feats[i].unsqueeze(-1)
            # 假設 ViT-B/32
            sim_map = sim.squeeze(-1).cpu().numpy().reshape(7, 7)

            # 顯示
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.imshow(sim_map, cmap="jet", alpha=0.5, extent=(0, w, h, 0))
            plt.axis("off")
            plt.title(
                f"[{sample['video_id']}] Prompt: \"{prompt}\"\nTrue: {sample['label']} | Pred: {sample['pred']} | Score: {sample['score']:.2f}")
            plt.show()
