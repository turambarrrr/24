import os
import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import ReIDNet

# 图库路径（可通过命令行参数或直接修改）
GALLERY_PATH = "gallery"   # 你可以改为任意路径，运行时会生成该目录下的缓存文件

CACHE_FILE = os.path.join(GALLERY_PATH, "gallery_features.pkl")
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 加载模型权重，自动获取类别数
state_dict = torch.load("best_model.pth", map_location=DEVICE)
num_classes = state_dict['fc.weight'].shape[0]
print(f"从模型权重中读取类别数: {num_classes}")

model = ReIDNet(num_classes=num_classes).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

image_paths = [os.path.join(GALLERY_PATH, f) for f in os.listdir(GALLERY_PATH)
               if f.lower().endswith(('.jpg','.jpeg','.png'))]
print(f"找到 {len(image_paths)} 张图片")

def extract_batch(paths, batch_size):
    all_feats = []
    all_names = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i+batch_size]
        batch_imgs = []
        valid_names = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img_t = transform(img)
                batch_imgs.append(img_t)
                valid_names.append(os.path.basename(p))
            except Exception as e:
                print(f"跳过 {p}: {e}")
        if not batch_imgs:
            continue
        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            _, feats = model(batch_tensor)
        all_feats.append(feats.cpu().numpy())
        all_names.extend(valid_names)
    if not all_feats:
        return np.array([]), []
    return np.vstack(all_feats), all_names

feats, names = extract_batch(image_paths, BATCH_SIZE)
print(f"特征矩阵形状: {feats.shape}")

# 保存到图库目录下
with open(CACHE_FILE, "wb") as f:
    pickle.dump({"feats": feats, "names": names}, f)
print(f"特征缓存已保存到 {CACHE_FILE}")