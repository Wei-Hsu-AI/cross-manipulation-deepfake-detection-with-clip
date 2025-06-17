from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import os
from PIL import Image
from sklearn.model_selection import train_test_split


class FrameDataset(Dataset):
    def __init__(self, root_dir, label, processor):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True) +
            glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        )
        if len(self.image_paths) == 0:
            print(f"[WARNING] No images found in {root_dir}")
        self.label = label
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"][0]
        return {"image": pixel_values, "label": self.label, "path": img_path}


def build_dataloaders(data_root, processor, batch_size=32, val_ratio=0.2):
    video_dirs = sorted([
        os.path.join(data_root, "Real_youtube", d)
        for d in os.listdir(os.path.join(data_root, "Real_youtube"))
        if os.path.isdir(os.path.join(data_root, "Real_youtube", d))
    ])
    train_dirs, val_dirs = train_test_split(
        video_dirs, test_size=val_ratio, random_state=42)

    def collect_frames(video_dirs, label):
        return ConcatDataset([FrameDataset(d, label, processor) for d in video_dirs])

    train_real_dataset = collect_frames(train_dirs, 0)
    val_real_dataset = collect_frames(val_dirs, 0)
    fake_train_dataset = FrameDataset(
        os.path.join(data_root, "FaceSwap"), 1, processor)
    fake_val_dataset = FrameDataset(os.path.join(
        data_root, "NeuralTextures"), 1, processor)

    train_dataset = ConcatDataset([train_real_dataset, fake_train_dataset])
    val_dataset = ConcatDataset([val_real_dataset, fake_val_dataset])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader
