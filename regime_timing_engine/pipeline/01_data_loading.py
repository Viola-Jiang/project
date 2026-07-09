"""
pipeline/01_data_loading.py
==============================
对应方法论文档 §2「数据与特征构建」的数据来源说明。

原先这一步是用 engine/synthetic_data.py 生成合成三区制数据；现在切换到
真实中证800数据，这一步变成单纯的"读取+标准化+画一张价格路径图"。

**占位格式说明**：真实数据文件尚未就位，本脚本按占位格式设计：
  输入: data/csi800.csv，至少包含 ['date', 'close'] 两列
  若拿到的真实文件列名不同（比如用的是'收盘价'或其它行情源字段名），
  只需改下面 RAW_COLUMNS 这一处映射，不需要改下游任何脚本。

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

RAW_PATH = DATA_DIR / "csi800.csv"
# 占位映射：原始文件的日期/收盘价列名 -> 标准化列名。真实数据到位后如果
# 列名不同，只需改这一处。
RAW_COLUMNS = {"date": "date", "close": "price"}


def load_raw_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到真实数据文件 {path}。请把中证800价格数据放到该路径，"
            f"至少包含 {list(RAW_COLUMNS.keys())} 这些列（列名不同的话改"
            f"本脚本顶部的 RAW_COLUMNS 映射即可）。"
        )
    df = pd.read_csv(path)
    df = df.rename(columns={raw: std for raw, std in RAW_COLUMNS.items()})
    missing = {"date", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} 标准化后仍缺少列: {missing}，请检查 RAW_COLUMNS 映射")
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "price"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def make_diagnostic_plot(df: pd.DataFrame, save_path: Path):
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df["price"], color="black", lw=1)
    ax.set_ylabel("指数点位")
    ax.set_xlabel("日期")
    ax.set_title("中证800价格路径（原始数据，尚未做特征工程/区制标注）")
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
