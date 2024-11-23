from typing import Tuple
import os
import argparse
import pandas as pd


def get_args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="source dataset", default="hrc_whu")
    parser.add_argument(
        "--target", type=str, help="target dataset", default="gf12ms_whu_gf1"
    )
    args = parser.parse_args()
    return args.source, args.target


def get_csv_data(source, target):
    file_path = os.path.join("logs", "eval_result", f"{source}_to_{target}_metrics.csv")
    df = pd.read_csv(file_path)
    source_target = df.to_numpy().tolist()
    return source_target


def make_latex(source: str, target: str):
    source_target_data = get_csv_data(source, target)
    target_source_data = get_csv_data(target, source)

    if source == "l8_biome_crop":
        source = "l8_biome"
    if target == "l8_biome_crop":
        target = "l8_biome"

    caption = (
        r"\caption{Comparison of different methods for domain adaptation between "
        + r"\_".join([s.upper() for s in source.split("_")])
        + r" and "
        + r"\_".join([s.upper() for s in target.split("_")])
        + r" datasets. The left side presents the results of adapting from "
        + r"\_".join([s.upper() for s in source.split("_")])
        + r" to "
        + r"\_".join([s.upper() for s in target.split("_")])
        + r", while the right side shows the results from "
        + r"\_".join([s.upper() for s in target.split("_")])
        + " to "
        + r"\_".join([s.upper() for s in source.split("_")])
        + r".}"
    )
    latex_code = (
        r"""\begin{table*}[ht]"""
        + "\n"
        + r"""\setlength{\tabcolsep}{0.9mm}"""
        + caption
        + "\n"
        + r"""\begin{tabular}{lccccccc|cccccccc}
    \toprule
    \multicolumn{8}{c|}"""
    )
    latex_code += (
        r"{\textbf{"
        + r"\_".join([s.upper() for s in source.split("_")])
        + r"$\rightarrow$"
        + r"\_".join([s.upper() for s in target.split("_")])
        + r"}}"
        + "\n"
    )

    latex_code += (
        r"& \multicolumn{8}{c}{\textbf{"
        + r"\_".join([s.upper() for s in target.split("_")])
        + r"$\rightarrow$"
        + r"\_".join([s.upper() for s in source.split("_")])
        + r"}}\\"
    )
    latex_code += "\n"
    latex_code += r"\midrule "

    latex_code += r"""
\multicolumn{1}{l|}{\textbf{Methods}} & \multicolumn{1}{l|}{\textbf{aAcc}} & \multicolumn{1}{l|}{\textbf{mIoU}} & \multicolumn{1}{l|}{\textbf{mAcc}} & \multicolumn{1}{l|}{\textbf{mDice}} & \multicolumn{1}{l|}{\textbf{mF1score}} & \multicolumn{1}{l|}{\textbf{mPrecision}} & \multicolumn{1}{l|}{\textbf{mRecall}} & \multicolumn{1}{l|}{\textbf{Methods}} & \multicolumn{1}{l|}{\textbf{aAcc}} & \multicolumn{1}{l|}{\textbf{mIoU}} & \multicolumn{1}{l|}{\textbf{mAcc}} & \multicolumn{1}{l|}{\textbf{mDice}} & \multicolumn{1}{l|}{\textbf{mF1score}} & \multicolumn{1}{l|}{\textbf{mPrecision}} & \multicolumn{1}{l}{\textbf{mRecall}} \\ 
\midrule
"""

    for data1, data2 in zip(source_target_data, target_source_data):
        for idx, item in enumerate(data1):
            if idx == 0:
                latex_code += r"\multicolumn{1}{l|}{" + f"{item}" + r"}" + "          "
            elif idx == len(data1) - 1:
                latex_code += (
                    r"& " + f"{item:.2f}" + "                                 "
                )
            else:
                latex_code += (
                    r"& \multicolumn{1}{c|}{" + f"{item:.2f}" + r"}" + "          "
                )
        for idx, item in enumerate(data2):
            if idx == 0:
                latex_code += (
                    r"& \multicolumn{1}{c|}{" + f"{item}" + r"}" + "          "
                )
            elif idx == len(data1) - 1:
                latex_code += r"& " + f"{item:.2f}" + "          "
            else:
                latex_code += (
                    r"& \multicolumn{1}{c|}{" + f"{item:.2f}" + r"}" + "          "
                )

        latex_code += r"\\"
    latex_code += r"\bottomrule"
    latex_code += r"""
\end{tabular}
\end{table*}
    """
    print(latex_code)


if __name__ == "__main__":
    # example usage:python src/tools/latex_for_cross.py --source hrc_whu --target gf12ms_whu_gf1
    source, target = get_args()
    make_latex(source, target)
#
