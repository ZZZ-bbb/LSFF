import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pytorch_msssim import ssim
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from best.unetFreqECA_deepconv import UNetWithFreqFECASelect_deepconv

# 类别颜色映射
COLOR_CLASS_MAPPING = {
    (250, 50, 83): 0,  # 穗
    (61, 61, 245): 1,  # 叶
    (0, 0, 0): 2  # 背景
}


class LargeImageDataset(Dataset):
    """大图像数据集类"""

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        print(f"\n=== 初始化数据集 ===")
        print(f"图像目录: {image_dir}")
        print(f"掩码目录: {mask_dir}")

        # 获取所有文件并验证
        self.images = []
        all_files = os.listdir(image_dir)
        print(f"\n目录中总文件数: {len(all_files)}")

        # 检查每个文件
        for filename in sorted(all_files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 构建对应的mask文件名
                mask_filename = filename.rsplit('.', 1)[0] + '.png'
                mask_path = os.path.join(mask_dir, mask_filename)

                if os.path.exists(mask_path):
                    self.images.append(filename)
                    print(f"成功匹配: {filename} -> {mask_filename}")
                else:
                    print(f"警告: 找不到对应的mask文件: {mask_filename}")

        print(f"\n成功加载的图像数量: {len(self.images)}")
        if len(self.images) == 0:
            raise RuntimeError("没有找到有效的图像-掩码对！")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name.rsplit('.', 1)[0] + '.png')

            # 读取图像和掩码
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")

            # 转换为numpy数组
            image_np = np.array(image)
            mask_np = np.array(mask)

            # 转换掩码为类别索引
            mask_class = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
            for color, class_idx in COLOR_CLASS_MAPPING.items():
                mask_class[(mask_np == color).all(axis=2)] = class_idx

            if self.transform:
                image_np = self.transform(image_np)

            return image_np, mask_class, img_name

        except Exception as e:
            print(f"处理文件出错 {img_name}: {str(e)}")
            raise


def apply_crf(image, pred_prob):
    """
    应用CRF后处理
    image: 原始图像 (H, W, 3)
    pred_prob: 预测概率 (C, H, W)
    """
    # 确保图像数据是连续的并且是uint8类型
    image = np.ascontiguousarray(image)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

        # 确保预测概率是连续的并且形状正确
    pred_prob = np.ascontiguousarray(pred_prob)

    h, w = image.shape[:2]

    # 创建CRF模型
    d = dcrf.DenseCRF2D(w, h, 3)

    # 将预测概率转换为unary potential
    U = unary_from_softmax(pred_prob)
    d.setUnaryEnergy(U)

    # 添加位置和颜色特征的势能项
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=15, rgbim=image, compat=18)

    # 进行推理
    Q = d.inference(5)

    # 返回最可能的类别
    return np.argmax(Q, axis=0).reshape((h, w))


def sliding_window_prediction(model, image, window_size=256, stride=128):
    """使用滑动窗口进行预测"""
    device = image.device
    h, w = image.shape[1:]
    pred_prob = torch.zeros((3, h, w), device=device)
    count = torch.zeros((1, h, w), device=device)

    # 收集所有图像块和其坐标
    patches = []
    coordinates = []
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patch = image[:, i:i + window_size, j:j + window_size].unsqueeze(0)
            patches.append(patch)
            coordinates.append((i, j))

            # 批量处理图像块
    batch_size = 2
    with torch.cuda.amp.autocast():
        for idx in range(0, len(patches), batch_size):
            batch_patches = torch.cat(patches[idx:idx + batch_size])
            with torch.no_grad():
                batch_outputs = model(batch_patches)

            for k, (i, j) in enumerate(coordinates[idx:idx + batch_size]):
                if k < len(batch_outputs):
                    pred_prob[:, i:i + window_size, j:j + window_size] += batch_outputs[k]
                    count[:, i:i + window_size, j:j + window_size] += 1

                    # 计算平均预测概率
    pred_prob = pred_prob / count

    # 将预测结果转移到CPU并应用softmax
    pred_prob = torch.softmax(pred_prob, dim=0).cpu().numpy()

    # 获取原始图像数据并进行预处理
    orig_image = image.cpu().numpy().transpose(1, 2, 0)

    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_image = std * orig_image + mean
    orig_image = np.clip(orig_image * 255, 0, 255).astype(np.uint8)

    try:
        # 应用CRF后处理
        final_pred = apply_crf(orig_image, pred_prob)
        return torch.from_numpy(final_pred).to(device)
    except Exception as e:
        print(f"CRF处理出错: {str(e)}")
        print(f"图像形状: {orig_image.shape}, 数据类型: {orig_image.dtype}")
        print(f"预测概率形状: {pred_prob.shape}, 数据类型: {pred_prob.dtype}")
        # 如果CRF失败，返回原始预测结果
        return torch.from_numpy(pred_prob.argmax(axis=0)).to(device)


def compute_iou(pred, target, num_classes):
    """计算IoU指标"""
    ious = torch.zeros(num_classes, device=pred.device)
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        ious[cls] = intersection / (union + 1e-10)
    return ious


def compute_ssim(pred, target, num_classes):
    """计算每个类别的SSIM"""
    ssim_scores = torch.zeros(num_classes, device=pred.device)

    for cls in range(num_classes):
        # 转换为二值掩码
        pred_cls = (pred == cls).float().unsqueeze(0).unsqueeze(0)  # Add channel and batch dims
        target_cls = (target == cls).float().unsqueeze(0).unsqueeze(0)

        # 计算SSIM
        score = ssim(pred_cls, target_cls, data_range=1.0, size_average=True)
        ssim_scores[cls] = score

    return ssim_scores


def compute_qseg(total_true_positives, total_false_positives, total_false_negatives):
    """计算Qseg指标"""
    qseg_scores = []
    for cls in range(3):
        tp = total_true_positives[cls]
        fp = total_false_positives[cls]
        fn = total_false_negatives[cls]
        qseg = tp / (tp + fp + fn + 1e-10)
        qseg_scores.append(qseg)
    return qseg_scores


def test_model():
    """模型测试主函数"""
    print("\n=== 开始测试过程 ===")

    # 设置数据转换
    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 验证目录
    image_dir = r'/'
    mask_dir = r'/'

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

        # 创建测试数据集和数据加载器
    test_dataset = LargeImageDataset(image_dir, mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"\n数据加载器初始化完成")
    print(f"总批次数: {len(test_loader)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = UNetWithFreqFECASelect_deepconv()
    checkpoint = torch.load('UNetWithFreqFECASelect_deepconv.pth', map_location=device)
    # 如果是完整的checkpoint
    # if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:  # 如果只保存了state_dict
    #     adjusted_state_dict = {key.replace('module.', ''): value
    #                            for key, value in checkpoint.items()}
    #     model.load_state_dict(adjusted_state_dict, strict=False)
    adjusted_state_dict = {key.replace('module.', ''): value
                           for key, value in checkpoint.items()}
    model.load_state_dict(adjusted_state_dict, strict=False)
    model.eval()


    print(f"使用设备: {device}")
    model.to(device)

    # 初始化评估指标
    processed_images = []
    total_ious = []
    total_ssims = []  # 添加SSIM跟踪

    print("\n=== 开始处理图像 ===")
    with torch.no_grad():
        for batch_idx, (image, mask, img_name) in enumerate(test_loader):
            print(f"\n处理图像 {batch_idx + 1}/{len(test_loader)}: {img_name[0]}")

            image = image.to(device)
            mask = mask.to(device)

            # 记录已处理的图像
            processed_images.append(img_name[0])

            # 使用滑动窗口预测
            pred = sliding_window_prediction(model, image[0])

            # 计算IoU指标
            ious = compute_iou(pred, mask[0], 3)
            miou = ious.mean().item()

            # 计算SSIM
            ssim_scores = compute_ssim(pred, mask[0], 3)
            mean_ssim = ssim_scores.mean().item()

            print(f"mIoU: {miou:.4f}")
            print(f"平均SSIM: {mean_ssim:.4f}")
            print(f"各类别IoU: 穗: {ious[0]:.4f}, 叶: {ious[1]:.4f}, 背景: {ious[2]:.4f}")
            print(f"各类别SSIM: 穗: {ssim_scores[0]:.4f}, 叶: {ssim_scores[1]:.4f}, 背景: {ssim_scores[2]:.4f}")

            total_ious.append(ious.cpu().numpy())
            total_ssims.append(ssim_scores.cpu().numpy())

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                # 打印最终统计信息
    print("\n=== 测试完成 ===")
    print(f"总共处理的图像数量: {len(processed_images)}")
    print("处理的图像列表:")
    for img in processed_images:
        print(f"- {img}")

        # 计算平均指标
    total_ious = np.array(total_ious)
    total_ssims = np.array(total_ssims)
    overall_miou = np.mean(total_ious)
    overall_class_ious = np.mean(total_ious, axis=0)
    overall_mssim = np.mean(total_ssims)
    overall_class_ssims = np.mean(total_ssims, axis=0)

    print(f"\n最终结果:")
    print(f"总体 mIoU: {overall_miou:.4f}")
    print(f"总体 SSIM: {overall_mssim:.4f}")
    print(f"各类别平均IoU:")
    print(f"- 穗: {overall_class_ious[0]:.4f}")
    print(f"- 叶: {overall_class_ious[1]:.4f}")
    print(f"- 背景: {overall_class_ious[2]:.4f}")
    print(f"各类别平均SSIM:")
    print(f"- 穗: {overall_class_ssims[0]:.4f}")
    print(f"- 叶: {overall_class_ssims[1]:.4f}")
    print(f"- 背景: {overall_class_ssims[2]:.4f}")


if __name__ == "__main__":
    test_model()