# extract_prw_gallery.py
import os
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

PRW_ROOT = 'PRW'
GALLERY_DIR = 'gallery'
os.makedirs(GALLERY_DIR, exist_ok=True)

# 读取训练集帧
train_mat = loadmat(os.path.join(PRW_ROOT, 'frame_train.mat'))
train_frames_data = train_mat['img_index_train'].flatten()
train_frames = set([str(f[0]) if isinstance(f, np.ndarray) else str(f) for f in train_frames_data])

all_frames = [f.replace('.jpg', '') for f in os.listdir(os.path.join(PRW_ROOT, 'frames')) if f.endswith('.jpg')]
test_frames = [f for f in all_frames if f not in train_frames]
print(f"测试集图片数: {len(test_frames)}")

for frame_name in tqdm(test_frames, desc="Extracting gallery"):
    img_path = os.path.join(PRW_ROOT, 'frames', frame_name + '.jpg')
    ann_path = os.path.join(PRW_ROOT, 'annotations', frame_name + '.jpg.mat')
    if not os.path.exists(img_path) or not os.path.exists(ann_path):
        continue

    try:
        ann = loadmat(ann_path)
        # 尝试多个可能的键名
        box_key = None
        for key in ['box_new', 'box', 'boxes']:
            if key in ann:
                box_key = key
                break
        if box_key is None:
            continue
        boxes_data = ann[box_key]
        if boxes_data.shape[1] == 5:
            boxes = boxes_data[:, :4].astype(np.int32)
            ids = boxes_data[:, 4].flatten().astype(np.int64)
        else:
            id_key = None
            for key in ['id', 'ids', 'pid', 'ID']:
                if key in ann:
                    id_key = key
                    break
            if id_key is None:
                continue
            boxes = boxes_data.astype(np.int32)
            ids = ann[id_key].flatten()
    except Exception as e:
        continue

    img = cv2.imread(img_path)
    for i, pid in enumerate(ids):
        if pid == -2:
            continue
        x, y, w, h = boxes[i]
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        out_name = f"{pid}_{frame_name}.jpg"
        out_path = os.path.join(GALLERY_DIR, out_name)
        cv2.imwrite(out_path, crop)

print(f"图库提取完成，共 {len(os.listdir(GALLERY_DIR))} 张图片")