import os
from PIL import Image
import numpy as np
from collections import defaultdict

def analyze_image_list(list_file):
    """
    分析图像列表文件中所有图像的属性
    
    Args:
        list_file: 包含图像路径的文本文件
    
    Returns:
        统计信息字典
    """
    stats = {
        'sizes': defaultdict(int),  # 记录不同尺寸的数量
        'widths': [],  # 所有宽度
        'heights': [],  # 所有高度
        'aspects': [],  # 所有宽高比
        'total': 0,  # 总图像数
    }
    
    base_dir = os.path.dirname(list_file)
    
    with open(list_file, 'r') as f:
        for line in f:
            # 分割行并获取图像路径
            parts = line.strip().split()
            if not parts:
                continue
                
            img_path = parts[0]
            if img_path.startswith('./'):
                img_path = img_path[2:]
            
            # 构建完整路径
            full_path = os.path.join(base_dir, img_path)
            
            try:
                with Image.open(full_path) as img:
                    width, height = img.size
                    aspect = width / height
                    
                    # 记录统计信息
                    size_key = f"{width}x{height}"
                    stats['sizes'][size_key] += 1
                    stats['widths'].append(width)
                    stats['heights'].append(height)
                    stats['aspects'].append(aspect)
                    stats['total'] += 1
            except Exception as e:
                print(f"处理图像 {full_path} 时出错: {e}")
                
    return stats

def print_statistics(stats, dataset_name):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
        dataset_name: 数据集名称
    """
    print(f"\n{dataset_name}统计信息:")
    print(f"总图像数: {stats['total']}")
    
    if stats['total'] > 0:
        # 计算尺寸统计
        widths = np.array(stats['widths'])
        heights = np.array(stats['heights'])
        aspects = np.array(stats['aspects'])
        
        print(f"\n分辨率统计:")
        print(f"宽度: 最小={widths.min()}, 最大={widths.max()}, 平均={widths.mean():.2f}")
        print(f"高度: 最小={heights.min()}, 最大={heights.max()}, 平均={heights.mean():.2f}")
        print(f"宽高比: 最小={aspects.min():.2f}, 最大={aspects.max():.2f}, 平均={aspects.mean():.2f}")
        
        print("\n最常见的图像尺寸:")
        common_sizes = sorted(stats['sizes'].items(), key=lambda x: x[1], reverse=True)
        for size, count in common_sizes[:5]:
            print(f"{size}: {count}张图像 ({count/stats['total']*100:.2f}%)")

def main():
    # 分析训练集
    basic_path = "/root/autodl-tmp/users/dhc/POSE/Classifier"
    train_list = os.path.join(basic_path, "train_list.txt")
    train_stats = analyze_image_list(train_list)
    print_statistics(train_stats, "训练集")
    
    # 分析测试集
    test_list = os.path.join(basic_path, "test_list.txt")
    test_stats = analyze_image_list(test_list)
    print_statistics(test_stats, "测试集")

if __name__ == "__main__":
    main()