# /root/autodl-tmp/Time-Series-Library/utils/tsfile_utils.py
import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe

def load_any_tsfile(filepath):
    """兼容所有格式的TS文件加载器"""
    try:
        # 尝试标准加载方式
        return load_from_tsfile_to_dataframe(filepath)
    except ValueError as e:
        if "could not convert string to float" in str(e):
            return _parse_with_brackets(filepath)
        raise

def _parse_with_brackets(filepath):
    """手动解析带括号格式"""
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 解析数据
    X, y = [], []
    for line in lines[7:]:  # 跳过文件头
        if ":" not in line:
            continue
            
        # 处理格式: (val1,val2,...) (val1,val2,...) :label
        channels, label = line.rsplit(":", 1)
        sample = []
        for ch in channels.split():
            values = ch.strip("()").split(",")
            sample.append([float(x) for x in values])
        X.append(np.array(sample).T)  # 转为(seq_len, n_channels)
        y.append(label)
    
    return pd.DataFrame(X), pd.Series(y)

def validate_tsfile(filepath):
    """验证TS文件格式"""
    print(f"验证文件: {filepath}")
    with open(filepath) as f:
        header = [next(f) for _ in range(7)]
        first_line = next(f).strip()
    
    print("文件头验证:")
    assert header[0].startswith("@problemName"), "缺少问题声明"
    assert header[2].startswith("@univariate false"), "必须为多变量"
    print("✅ 文件头验证通过")

    print("首行数据验证:", first_line[:50] + "...")
    assert ":" in first_line, "缺少标签分隔符"
    assert first_line.count("(") == first_line.count(")"), "括号不匹配"
    print("✅ 数据行基础验证通过")
    
    # 数值格式检查
    channels = first_line.split(":")[0].split()
    for ch in channels[:3]:  # 检查前3个通道
        values = ch.strip("()").split(",")[:3]  # 检查前3个值
        for v in values:
            try:
                float(v)
            except:
                raise ValueError(f"非法数值格式: {v}")
    print("✅ 数值格式验证通过")