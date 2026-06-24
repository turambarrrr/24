import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PRWDataset(Dataset):
    def __init__(self, root='PRW', split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform if transform else self._default_transform()

        # 获取帧列表：训练集使用 frame_train.mat，测试集使用差集
        if split == 'train':
            mat_path = os.path.join(root, 'frame_train.mat')
            data = loadmat(mat_path)
            frames_data = data['img_index_train'].flatten()
            frames = [str(f[0]) if isinstance(f, np.ndarray) else str(f) for f in frames_data]
        else:
            # 测试集：所有帧 - 训练集帧
            train_mat = loadmat(os.path.join(root, 'frame_train.mat'))
            train_frames_data = train_mat['img_index_train'].flatten()
            train_frames = set([str(f[0]) if isinstance(f, np.ndarray) else str(f) for f in train_frames_data])
            all_frames = [f.replace('.jpg', '') for f in os.listdir(os.path.join(root, 'frames')) if f.endswith('.jpg')]
            frames = [f for f in all_frames if f not in train_frames]

        self.frames = frames
        self.data = []
        for frame_name in self.frames:
            ann_file = os.path.join(root, 'annotations', frame_name + '.jpg.mat')
            if not os.path.exists(ann_file):
                continue
            ann = loadmat(ann_file)
            if 'box_new' not in ann:
                continue
            boxes = ann['box_new']  # shape (N,5), 每行: [id, x, y, w, h]
            img_path = os.path.join(root, 'frames', frame_name + '.jpg')
            for row in boxes:
                pid, x, y, w, h = row   # 注意顺序：id 在第一列
                pid = int(pid)
                if pid == -2:
                    continue
                # 确保坐标整数
                x, y, w, h = int(x), int(y), int(w), int(h)
                self.data.append((img_path, pid, x, y, w, h))

        self.id_list = sorted(set([pid for _, pid, _, _, _, _ in self.data]))
        self.id2label = {pid: i for i, pid in enumerate(self.id_list)}
        print(f"PRW {split} set: {len(self.data)} samples, {len(self.id_list)} IDs")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),   # 轻微旋转，保留方向信息
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, x, y, w, h = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        box = (x, y, x+w, y+h)
        cropped = img.crop(box)
        img_t = self.transform(cropped)
        label = self.id2label[pid]
        return img_t, label