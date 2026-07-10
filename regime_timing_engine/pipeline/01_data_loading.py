"""
pipeline/01_data_loading.py
==============================
§2「数据与特征构建」

这一步基于真实中证800全收益指数数据，实现：
“读取真实数据 → 标准化成内部统一格式 → 画一张原始价格图存档。”

数据来源：data/csi800_total_return.xlsx —— 中证800全收益指数（纳入
分红再投资）日频收盘点位。原始文件是 Sheet1 两列、无表头：[日期, 收盘点位]，
2009-01-05 起。

运行方式：
  python pipeline/01_data_loading.py
输出：
  data/prices.csv （标准化列：date, price）
  outputs/figures/01_data_loading.png
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
sys.path.insert(0, str(REPO_ROOT))

from engine.plotting import setup_cjk_font  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RAW_PATH = DATA_DIR / "csi800_total_return.xlsx"


def load_raw_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到真实数据文件 {path}。请把中证800全收益指数数据（Sheet1，"
            f"两列[日期, 收盘点位]，无表头）放到该路径。"
        )
    df = pd.read_excel(path, sheet_name=0, header=None, names=["date", "price"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def make_diagnostic_plot(df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df["price"], color="black", lw=1)
    ax.set_ylabel("指数点位")
    ax.set_xlabel("日期")
    ax.set_title("中证800价格路径")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"诊断图已保存 -> {save_path}")


if __name__ == "__main__":
    df = load_raw_prices(RAW_PATH)

    out_path = DATA_DIR / "prices.csv"
    df.to_csv(out_path, index=False)
    print(f"已加载 {len(df)} 个交易日的价格数据 -> {out_path}")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    make_diagnostic_plot(df, FIGURES_DIR / "01_data_loading.png")
