"""S2 零损失验证：FeatureStore 物化切片 vs LightGBM DA feature_engineering 现状重算，逐位 diff。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
import numpy as np
import pandas as pd
from utils.feature_store import FeatureStore
from utils.resolution import resolve_resolution

SOURCE = r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\data\shandong_pmos_96_full_v2.xlsx"

# 1. FeatureStore 物化
fs = FeatureStore(resolution="15min", source=SOURCE).ensure()
fs_cols = set(fs.get_feature_columns())
print(f"FeatureStore 物化完成: {fs.version}, {len(fs._da)} 行")
print(f"特征列: {len(fs_cols)} 个")

# 2. 现状 LightGBM feature_engineering 重算基准
from lightGBM.infer_da_fix import PowerInference
inf = PowerInference(model_path=None, resolution=resolve_resolution("15min"))
df = pd.read_excel(SOURCE, engine="openpyxl")
df["ds"] = pd.to_datetime(df["时刻"], errors="coerce")
df["y"] = pd.to_numeric(df["日前电价"], errors="coerce")
df["load"] = pd.to_numeric(df["直调负荷预测值"], errors="coerce").ffill()
df["wind"] = pd.to_numeric(df["风电总加预测值"], errors="coerce").ffill()
df["solar"] = pd.to_numeric(df["光伏总加预测值"], errors="coerce").ffill()
df["interconnect"] = pd.to_numeric(df["联络线受电负荷预测值"], errors="coerce").ffill()
df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
baseline = inf.feature_engineering(df, resolution=resolve_resolution("15min"))

# 3. 对齐比较：FeatureStore 切片 vs baseline
fs_df = fs.slice_da("2026-06-01").copy()
base_df = baseline.copy()

# 特征列交集
common_cols = sorted(fs_cols & set(base_df.columns))
print(f"共同特征列: {len(common_cols)} 个")

# 按 ds 对齐
fs_df = fs_df.set_index("ds")
base_df = base_df.set_index("ds")
common_idx = fs_df.index.intersection(base_df.index)
print(f"对齐行数: {len(common_idx)}")

# 逐位比较
max_diff = 0.0
n_mismatch = 0
mismatch_cols = {}
for col in common_cols:
    a = fs_df.loc[common_idx, col]
    b = base_df.loc[common_idx, col]
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    diff = np.abs(a - b)
    md = float(diff.max()) if len(diff) else 0.0
    max_diff = max(max_diff, md)
    if md > 1e-8:
        n_mismatch += 1
        mismatch_cols[col] = md
        print(f"  ⚠ {col}: max_diff={md:.6f}")

print(f"\n=== 零损失验证 ===")
print(f"最大绝对差: {max_diff:.10f}")
print(f"不一致特征列: {n_mismatch}/{len(common_cols)}")
if n_mismatch == 0 and max_diff < 1e-8:
    print("✅ PASS: FeatureStore 物化切片与现状特征工程逐位一致")
else:
    print("❌ FAIL: 存在差异，需排查 (见上方 mismatch_cols)")
    if mismatch_cols:
        print("差异列:", list(mismatch_cols.keys()))
