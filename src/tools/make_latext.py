import json
import argparse

def latex_code(filepath):

    res = r"""\begin{table*}[ht]
    \centering
    \caption{Performance Metrics for Different Models and Scenes}
    \begin{adjustbox}{width=\textwidth,height=\textheight,keepaspectratio}
    \begin{tabular}{llccccccc}
    \toprule
    \multirow{2}{*}{Scene} & \multirow{2}{*}{Method} & \multicolumn{7}{c}{Metrics} \\
    \cmidrule(lr){3-9}
    & & aAcc & mIoU & mAcc & mDice & mFscore & mPrecision & mRecall \\
    \midrule
    """

    with open(filepath, "r", encoding="utf-8") as file:
        scene_metrics = json.load(file)

    for scene in list(scene_metrics.keys()):
        res += r"\multirow{8}{*}{" + scene + "}\n"
        for model_name in list(scene_metrics[scene].keys()):
            res += f"& {model_name} "
            for metrics_name in list(scene_metrics[scene][model_name].keys()):
                val = scene_metrics[scene][model_name][metrics_name]
                res += f"& {val:.2f} "
            res += r"\\"
            res += "\n"
        res += r"\midrule"
        res += "\n"
    res += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{adjustbox}" + "\n" +r"\end{table*}"
    return res

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--filepath",type=str,required=True)
    args = parse.parse_args()
    return args.filepath

if __name__ == "__main__":
    # example usage: python src/tools/make_latext.py --filepath result.json
    filepath = get_args()
    latex = latex_code(filepath)
    print(latex)
