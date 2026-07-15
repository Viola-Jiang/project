"""
regime_engine/plotting.py
==========================
公共绘图工具：统一处理中文字体加载与常用配色。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REGIME_COLORS = {"bull": "#d94f4f", "sideways": "#e8b84b", "bear": "#3f6fa8"}

_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # Windows (常见安装路径)
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]


def setup_cjk_font():
    """加载中文字体。返回是否加载成功。"""
    for path in _FONT_CANDIDATES:
        try:
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


def shade_regimes(ax, df, regime_col="regime", date_col="date", alpha=0.12):
    """按连续区制段给坐标轴背景上色，date_col/regime_col需已在df中。"""
    seg_id = (df[regime_col] != df[regime_col].shift(1)).cumsum()
    for _, seg in df.groupby(seg_id):
        ax.axvspan(seg[date_col].iloc[0], seg[date_col].iloc[-1],
                   color=REGIME_COLORS.get(seg[regime_col].iloc[0], "gray"),
                   alpha=alpha, lw=0)
