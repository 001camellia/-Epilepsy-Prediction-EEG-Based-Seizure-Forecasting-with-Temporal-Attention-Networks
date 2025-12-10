import os
import sys
from data_provider.data_loader import UEAloader

def test_load():
    # 必须提供的参数
    root_path = "/root/autodl-tmp/Time-Series-Library/dataset/EthanolConcentration"
    args = {
        'data': 'UEA',
        'root_path': root_path,
        'data_path': 'EthanolConcentration_TRAIN.ts',
        'target': 'OT',
        'features': 'M'
    }
    
    print(f"\n=== 调试信息 ===")
    print(f"路径验证: {os.path.exists(root_path)}")
    print(f"文件列表: {os.listdir(root_path)}")
    
    # 初始化时需要传入参数
    loader = UEAloader(args)
    df, labels = loader.load_all(root_path)
    print(f"\n✅ 加载成功！样本数: {len(df)}, 标签数: {len(labels)}")

if __name__ == "__main__":
    test_load()