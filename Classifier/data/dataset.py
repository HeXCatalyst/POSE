import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 高级预处理策略
def resize_with_strategy(image, target_size=256):
    height, width = image.shape[:2]
    current_size = width  # 因为是正方形图像，宽高相等
    
    if current_size < target_size:
        # 对于低分辨率图像，使用双线性插值
        resized = cv2.resize(image, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
    
    elif current_size > target_size:
        # 对于高分辨率图像，使用金字塔向下采样
        current = image.copy()
        while current_size >= target_size * 2:
            current = cv2.pyrDown(current)
            current_size = current.shape[0]
        
        # 最后一步resize到目标尺寸
        resized = cv2.resize(current, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
    
    else:
        # 对于已经是256x256的图像，保持不变
        resized = image.copy()
    
    return resized

class ImageDataset(Dataset):
    """
    图像数据集类，支持高级预处理策略
    """
    def __init__(self, txt_path, transform=None, root_dir=None, target_size=256, use_advanced_resize=True):
        self.img_list = []
        self.label_list = []
        self.transform = transform
        self.root_dir = root_dir
        self.target_size = target_size
        self.use_advanced_resize = use_advanced_resize
        
        # 读取txt文件
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 分割路径和标签
            parts = line.split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                
                # 如果提供了根目录，则将相对路径转换为绝对路径
                if root_dir is not None:
                    if img_path.startswith('./'):
                        img_path = img_path[2:]  # 移除开头的 './'
                    img_path = os.path.join(root_dir, img_path)
                
                self.img_list.append(img_path)
                self.label_list.append(label)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.label_list[idx]
        
        # try:
        if self.use_advanced_resize:
            # 使用OpenCV读取图像并进行高级预处理
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # BGR转RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用resize策略
            processed = resize_with_strategy(image, self.target_size)
            
            # 标准化到[0,1]范围
            processed = processed.astype(np.float32) / 255.0
            
            # 转换为tensor
            processed = torch.from_numpy(processed).permute(2, 0, 1)
            
            # 如果有其他转换，继续应用
            if self.transform:
                processed = self.transform(processed)

            return processed, label
        else:
            # 使用原始的PIL和transforms处理流程
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

def get_data_loaders(train_txt, test_txt, batch_size=32, num_workers=4, root_dir=None):
    # 设置文件路径
    # 使用高级预处理策略创建数据集
    train_dataset = ImageDataset(
        train_txt,
        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
        root_dir=root_dir,
        target_size=256,
        use_advanced_resize=True
    )
    test_dataset = ImageDataset(
        test_txt,
        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
        root_dir=root_dir,
        target_size=256,
        use_advanced_resize=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def test():
    # 设置文件路径
    file_dir = os.path.dirname(os.path.abspath(__file__))
    basic_path = os.path.abspath(file_dir + os.path.sep + "..")
    train_txt = os.path.join(basic_path, "train_list.txt")
    test_txt = os.path.join(basic_path, "test_list.txt")

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        train_txt=train_txt,
        test_txt=test_txt,
        batch_size=32,
        root_dir=basic_path
    )

    # 打印数据集信息
    print(f"训练集-大小: {len(train_loader.dataset)} 样本")
    print(f"测试集-大小: {len(test_loader.dataset)} 样本")

    # 获取一个批次的样本并打印形状
    for images, labels in train_loader:
        print(f"训练集-批次图像形状: {images.shape}")
        print(f"训练集-批次标签形状: {labels.shape}")
        break
    for images, labels in test_loader:
        print(f"测试集-批次图像形状: {images.shape}")
        print(f"测试集-批次标签形状: {labels.shape}")
        break

if __name__ == "__main__":
    test()

