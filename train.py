import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from tqdm import tqdm
from unetFreqECAdeepconv import UNetWithFreqFECASelect_deepconv
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

COLOR_CLASS_MAPPING = {
    (250, 50, 83): 0,  # Ear
    (61, 61, 245): 1,  # Leaf
    (0, 0, 0): 2  # Background
}


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        mask_np = np.array(mask)
        mask_class = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        for color, class_idx in COLOR_CLASS_MAPPING.items():
            mask_class[(mask_np == color).all(axis=2)] = class_idx

        mask_tensor = torch.from_numpy(mask_class)

        return image, mask_tensor


def calculate_metrics(pred, target, num_classes=3):
    with torch.no_grad():
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        pixel_acc = accuracy_score(target.reshape(-1), pred.reshape(-1))
        precision = precision_score(target.reshape(-1), pred.reshape(-1),
                                    average='macro', zero_division=0)
        recall = recall_score(target.reshape(-1), pred.reshape(-1),
                              average='macro', zero_division=0)
        f1 = f1_score(target.reshape(-1), pred.reshape(-1),
                      average='macro', zero_division=0)

        return pixel_acc, precision, recall, f1


def calculate_miou(pred, target, num_classes):
    with torch.no_grad():
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds & target_inds).float().sum()
            union = (pred_inds | target_inds).float().sum()

            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append((intersection / union).item())

        return np.nanmean(ious), ious


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_pixel_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    pbar = tqdm(train_loader, desc="Training")
    start_time = time.time()

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                    mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        pixel_acc, precision, recall, f1 = calculate_metrics(preds, masks)

        total_pixel_acc += pixel_acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    epoch_time = time.time() - start_time
    num_batches = len(train_loader)

    return (total_loss / num_batches, epoch_time,
            total_pixel_acc / num_batches,
            total_precision / num_batches,
            total_recall / num_batches,
            total_f1 / num_batches)


def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_iou = 0
    class_ious = [0] * num_classes
    total_pixel_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    pbar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                        mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            miou, ious = calculate_miou(preds, masks, num_classes)
            pixel_acc, precision, recall, f1 = calculate_metrics(preds, masks)

            total_iou += miou
            for i, iou in enumerate(ious):
                if not np.isnan(iou):
                    class_ious[i] += iou

            total_pixel_acc += pixel_acc
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{miou:.4f}'})

    num_batches = len(val_loader)
    return (total_loss / num_batches,
            total_iou / num_batches,
            [iou / num_batches for iou in class_ious],
            total_pixel_acc / num_batches,
            total_precision / num_batches,
            total_recall / num_batches,
            total_f1 / num_batches)


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    train_full_dataset = CustomDataset(
        r'D:\yield\U-net\pythonProject2\data\slide_train\data-2\image_train',
        r'D:\yield\U-net\pythonProject2\data\slide_train\data-2\mask_train',
        transform=transform
    )

    test_dataset = CustomDataset(
        r'D:\yield\U-net\pythonProject2\data\slide_train\data-2\test\image_test',
        r'D:\yield\U-net\pythonProject2\data\slide_train\data-2\test\mask_test',
        transform=transform
    )

    # 分割训练集和验证集
    train_size = int(0.8 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    num_classes = 3
    model = UNetWithFreqFECASelect_deepconv(n_channels=3, n_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = CrossEntropyLoss2d()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )

    num_epochs = 300
    best_test_miou = 0
    best_val_miou = 0  # 新增：跟踪最佳验证集mIoU
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_mious = []
    test_mious = []
    class_ious = [[] for _ in range(num_classes)]

    # 开始训练
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        train_loss, epoch_time, train_pixel_acc, train_precision, train_recall, train_f1 = train(
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_miou, epoch_class_ious, val_pixel_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device, num_classes
        )
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            print(f"New best validation mIoU: {best_val_miou:.4f}, saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_miou': best_val_miou,
            }, 'kernel_size_17.pth')

            # 在epoch>=30后开始测试
        if epoch >= 300:  # 从第30个epoch开始测试（索引从0开始）
            print("\nRunning test evaluation...")
            test_loss, test_miou, test_class_ious, test_pixel_acc, test_precision, test_recall, test_f1 = validate(
                model, test_loader, criterion, device, num_classes
            )
            test_mious.append(test_miou)

            # 如果测试集上的mIoU更好，则保存模型
            if test_miou > best_test_miou:
                best_test_miou = test_miou
                print(f"New best test mIoU: {best_test_miou:.4f}, saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_miou': best_test_miou,
                }, 'kernel_size_17_test.pth')

            print(f"Test Results - mIoU: {test_miou:.4f}, Best Test mIoU: {best_test_miou:.4f}")
            print(f"Test Metrics - Pixel Acc: {test_pixel_acc:.4f}, Precision: {test_precision:.4f}, "
                  f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

            # 更新学习率
        scheduler.step(val_miou)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        for i, iou in enumerate(epoch_class_ious):
            class_ious[i].append(iou)

            # 打印当前epoch的训练情况
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_miou:.4f}")
        print(f"Train Metrics - Pixel Acc: {train_pixel_acc:.4f}, Precision: {train_precision:.4f}, "
              f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val Metrics - Pixel Acc: {val_pixel_acc:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # 训练结束后，加载最佳模型并进行最终测试
    print("\nTraining completed. Loading best model for final test...")
    checkpoint = torch.load('best_model_test.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_test_loss, final_test_miou, final_test_class_ious, final_test_pixel_acc, final_test_precision, final_test_recall, final_test_f1 = validate(
        model, test_loader, criterion, device, num_classes
    )

    print(f"\nFinal Test Results:")
    print(f"Best Test mIoU: {best_test_miou:.4f}")
    print(f"Class IoUs: {[f'{iou:.4f}' for iou in final_test_class_ious]}")
    print(f"Pixel Accuracy: {final_test_pixel_acc:.4f}")
    print(f"Precision: {final_test_precision:.4f}")
    print(f"Recall: {final_test_recall:.4f}")
    print(f"F1 Score: {final_test_f1:.4f}")


if __name__ == '__main__':
    main()