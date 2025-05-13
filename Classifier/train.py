import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data.dataset import get_data_loaders
from torchvision.models import ResNet50_Weights
import time
import os
import copy
import matplotlib.pyplot as plt
import numpy as np

# 获取当前目录
basic_path = os.path.dirname(os.path.abspath(__file__))


def train_model(model, train_loader, test_loader, num_epochs=20, device='cuda:0'):
    """
    训练模型函数
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数，默认20轮
        device: 训练设备，默认cuda:0
    Returns:
        model: 训练后的模型
    """
    # 将模型移至指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # 记录最佳模型和最佳准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 计算训练集上的平均损失和准确率
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 评估模式
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 统计
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # 计算测试集上的平均损失和准确率
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        
        # 如果当前模型性能最佳，保存模型权重
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # 保存最佳模型
            torch.save(best_model_wts, os.path.join(basic_path, "weights/best_model.pth"))
            print(f"保存最佳模型，准确率: {best_acc:.4f}")
        
        # 每2轮保存一次权重
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join(basic_path, f"weights/epoch_{epoch+1}.pth"))
            print(f"保存第 {epoch+1} 轮模型权重")
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # 训练结束，加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def main():
    # GPU or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # path to txt
    train_txt = os.path.join(basic_path, "train_list.txt")  # 训练数据
    test_txt = os.path.join(basic_path, "test_list.txt")  # 测试数据
    
    # data loader
    train_loader, test_loader = get_data_loaders(
        train_txt=train_txt,
        test_txt=test_txt,
        batch_size=128, 
        num_workers=8,
        root_dir=basic_path
    )
    
    # load pre-trained model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # 修改最后一层以适应多分类任务，并添加dropout
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # add dropout layer
        nn.Linear(num_features, 11)
    )
    
    # 确保权重保存目录存在
    weights_dir = os.path.join(basic_path, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,  # 修改为20轮
        device=device
    )
    print("训练完成")

if __name__ == "__main__":
    main()

