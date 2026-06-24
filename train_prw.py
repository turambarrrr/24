import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from prw_dataset import PRWDataset
from model import ReIDNet, TripletLoss
from tqdm import tqdm

if __name__ == '__main__':
    # ====== 强制 GPU ======
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        print("✅ 使用 GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("⚠️ 使用 CPU")

    # ====== 超参数 ======
    batch_size = 64
    epochs = 30
    lr_backbone = 0.0001
    lr_fc = 0.001
    val_ratio = 0.1
    margin = 0.2
    triplet_weight = 1.0          # 可调，平衡分类损失和三元组损失

    # ====== 加载数据集 ======
    print("加载数据集...")
    full_dataset = PRWDataset(root='PRW', split='train')
    num_classes = len(full_dataset.id_list)

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"训练集样本数: {train_size}, 验证集样本数: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ====== 模型：冻结前2层 (conv1, bn1, layer1, layer2) ======
    model = ReIDNet(num_classes, pretrained=True, freeze_layers=2).to(device)
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = TripletLoss(margin=margin)

    # ====== 差异化学习率 ======
    backbone_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': fc_params, 'lr': lr_fc}
    ], weight_decay=5e-4)

    # ====== 余弦退火学习率 ======
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    print("开始训练...")

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, feats = model(imgs)
            loss = ce_loss(logits, labels) + triplet_weight * triplet_loss(feats, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                loss = ce_loss(logits, labels) + triplet_weight * triplet_loss(feats, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR_b: {current_lrs[0]:.6f}, LR_fc: {current_lrs[1]:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), "simple_reid_model.pth")
    print("🎉 训练完成，最终模型已保存为 simple_reid_model.pth，最佳模型为 best_model.pth")