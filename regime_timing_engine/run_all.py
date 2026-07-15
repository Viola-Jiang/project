"""
run_all.py — 一键运行全部流程

运行顺序：
  阶段1: 01_data_loading          → data/prices.csv
  阶段2: 02_feature_engineering   → data/features.csv
  阶段3: validation（4个，emission → duration_hazard → bocpd → regime_assignment）
  阶段4: ablation（s0 → s1 → s2 → s3 → s4 → lookahead → leverage → backtest_report → summary）

"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    # (阶段名, 脚本路径, 是否必须成功)
    ("阶段1: 数据加载",       "ablation/01_data_loading.py",           True),
    ("阶段1: 特征工程",       "ablation/02_feature_engineering.py",    True),
    # validation
    ("阶段2: 发射模型验证",   "validation/emission_validation.py",          False),
    ("阶段2: 久期模型验证",   "validation/duration_hazard_validation.py",   False),
    ("阶段2: BOCPD引擎验证",  "validation/bocpd_validation.py",             False),
    ("阶段2: 区制分配验证",   "validation/regime_assignment_validation.py", False),
    # ablation — run_ablation_summary 内部会调 S0-S4 的 generate_positions()，
    # 所以不再单独跑独立脚本（重复跑浪费时间）。如需单独调试某个方案，
    # 可以在 PyCharm 里直接打开该脚本 Run。
    ("阶段3: 前视偏差对照",   "ablation/lookahead_contrast.py",       False),
    ("阶段3: 杠杆对照",       "ablation/leverage_contrast.py",        False),
    ("阶段3: 回测报告",       "ablation/backtest_report.py",          False),
    ("阶段3: 消融汇总(S0-S4)", "ablation/run_ablation_summary.py",    False),
]


def run_script(script_path: str) -> bool:
    """运行单个脚本，返回是否成功"""
    full_path = PROJECT_DIR / script_path
    print(f"\n{'='*60}")
    print(f"▶ 运行: {script_path}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_DIR),
        capture_output=False,  # 实时输出到终端
    )

    if result.returncode == 0:
        print(f"✅ {script_path} 完成")
        return True
    else:
        print(f"❌ {script_path} 失败 (退出码: {result.returncode})")
        return False


def main():
    print("=" * 60)
    print("  指数择时引擎 — 全流程运行")
    print(f"  工作目录: {PROJECT_DIR}")
    print("=" * 60)

    failed = []
    blocked = False

    for step_name, script_path, critical in STEPS:
        if blocked:
            print(f"\n⏭ 跳过: {step_name} — {script_path} (上游关键步骤失败)")
            failed.append((step_name, script_path))
            continue

        print(f"\n{'─'*60}")
        print(f"  {step_name}")
        print(f"{'─'*60}")

        ok = run_script(script_path)
        if not ok:
            failed.append((step_name, script_path))
            if critical:
                print(f"\n⚠️  关键步骤失败，后续步骤将被跳过")
                blocked = True

    # 汇总
    print("\n" + "=" * 60)
    print("  全流程结束")
    print("=" * 60)

    if not failed:
        print("✅ 全部步骤执行成功")
        print("\n输出文件:")
        print("  ablation 结果: outputs/ablation/results/*.csv")
        print("  ablation 图片: outputs/ablation/figures/*.png")
        print("  validation 结果: outputs/validation/results/*.csv")
        print("  validation 图片: outputs/validation/figures/*.png")
    else:
        print(f"❌ {len(failed)}/{len(STEPS)} 个步骤失败:")
        for name, path in failed:
            print(f"    • {name}: {path}")
        print("\n可单独重新运行失败的脚本")


if __name__ == "__main__":
    main()
