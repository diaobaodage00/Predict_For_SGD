import pandas as pd
from pathlib import Path

def fix_large_sgd(csv_path: Path, threshold: float = 456.0, scale: float = 100.0):
    df = pd.read_csv(csv_path, dtype=str)
    df['SGD'] = pd.to_numeric(df['SGD'], errors='coerce')
    mask = df['SGD'].notna() & (df['SGD'] >= threshold)
    if mask.sum() == 0:
        print("未找到满足条件的行（SGD >= {}).".format(threshold))
        return
    # 备份原文件
    bak_path = csv_path.with_suffix('.backup.csv')
    df.to_csv(bak_path, index=False, float_format='%.6f')
    # 修改并写回原文件
    df.loc[mask, 'SGD'] = df.loc[mask, 'SGD'] / scale
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"已处理 {mask.sum()} 行（SGD >= {threshold}），除以 {scale}。")
    print("备份文件：", bak_path)
    print("处理后的示例行：")
    print(df[mask].head(10))

if __name__ == "__main__":
    base = Path(__file__).parent
    csv_file = base / "data_subset.csv"
    fix_large_sgd(csv_file)