import pandas as pd
import os, sys
from itertools import islice


def summarize_attributes_importance(path, db_name, discretization_type, model):
    data = pd.read_csv(path)
    data.set_index('Unnamed: 0', inplace=True)
    data.index.name = None
    importance = {}
    N = data.shape[0]
    for col in data.columns:
        sum = 0
        for i in data[col]:
            sum += abs(i)
        importance[col] = sum / N
    sorted_importance_attributes = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    first_importance_attributes = dict(islice(sorted_importance_attributes.items(), 15))

    bip_attributes = []
    mod_attributes = []

    columns = []
    for col in data.columns:
        columns.append(col)

    for col in columns:
        if col.find('_bip') != -1:
            bip_attributes.append(col)

        elif col.find('_mod') != -1:
            mod_attributes.append(col)

    cols = []
    for key, value in first_importance_attributes.items():
        cols.append(key)

    latex_code = r"\begin{tikzpicture}" + "\n"
    latex_code += r"\begin{axis}[" + "\n"
    latex_code += r"xbar = -0.175cm," + "\n"
    latex_code += r"enlarge y limits=0.05," + "\n"
    latex_code += r"symbolic y coords={"+ r"{}".format(', '.join(map(str, cols)))+ r"}," + "\n"
    latex_code += r"ytick={" + r"{}".format(', '.join(map(str, cols)))+ r"}," + "\n"
    latex_code += r"y dir = reverse," + "\n"
    latex_code += r"bar width = 5pt," + "\n"
    latex_code += r"height=6.5cm," + "\n"
    latex_code += r"width=4.5cm," + "\n"
    latex_code += r"xmin=0," + "\n"
    latex_code += r"ticklabel style={font=\tiny},"
    # latex_code += r"ylabel={Features}," + "\n"
    latex_code += r"xlabel={" + r"{}".format(model) + r"}," + "\n"
    latex_code += r"legend pos = south east," + "\n"
    latex_code += r"legend style={" + "\n"
    # latex_code += r"font=\small,"  + "\n"
    # latex_code += r"legend cell align=left," + "\n"
    latex_code += r"nodes={scale=0.6}" + "\n"
    # latex_code += r"mark size=1pt," + "\n"
    latex_code += r"}" + "\n"
    latex_code += r"]" + "\n"

    latex_code += r"\addlegendimage{area legend, fill = blue}"+ "\n"
    latex_code += r"\addlegendentry{ord};" + "\n"
    latex_code += r"\addlegendimage{area legend, fill = red}" + "\n"
    latex_code += r"\addlegendentry{bip};" + "\n"
    latex_code += r"\addlegendimage{area legend, fill = green}" + "\n"
    latex_code += r"\addlegendentry{mod};" + "\n"

    for key, value in first_importance_attributes.items():
        cols.append(key)
        if key in bip_attributes:
            latex_code += r"\addplot[style={red, fill=red}] coordinates{" + r"({}".format(value) + "," + r"{})".format(key)+ r"};" + "\n"
        elif key in mod_attributes:
            latex_code += r"\addplot[style={green, fill=green}]  coordinates{" + r"({}".format(value) + "," + r"{})".format(key)+ r"};" + "\n"
        else:
            latex_code += r"\addplot[style={blue, fill=blue}]  coordinates{" + r"({}".format(
                value) + "," + r"{})".format(key) + r"};" + "\n"

    latex_code += r"\end{axis}" + "\n"
    latex_code += r"\end{tikzpicture}" + "\n"

    return latex_code

if __name__ == '__main__':
    # sys.excepthook = my_exception_hook
    args = sys.argv[1:]
    db_name = args[0]
    discretization_type = args[1]
    models1 = ["log", "svm", "lda"]
    models2 = ["rf", "xgb", "dtree"]
    global_code = ""

    # for submodels in models:
    #     global_code += r"\begin{figure}[H]" + "\n"
    #     global_code += r"\hspace{-1cm}" + "\n"
    #     for model in submodels:
    #         path = 'outputs/' + db_name.lower() + '/results/shap/' + discretization_type.lower() + '/shap_data_bip_mod_' + model + '.csv'
    #         global_code += r"\begin{minipage}{0.25\textwidth}" + "\n"
    #         global_code += summarize_attributes_importance(path, db_name, discretization_type, model)
    #         global_code += r"\end{minipage}"
    #         if model != submodels[1]:
    #             global_code += r"\hspace{2cm}"
    #     global_code += r"\caption{Features importance on " + db_name + " dataset case of " + discretization_type.lower() + "discretization}"
    #     global_code += r"\end{figure}"

    global_code += r"\begin{figure}[H]" + "\n"
    global_code += r"\hspace{-1cm}" + "\n"
    for model in models1:
        path = 'outputs/general_results/results/' + db_name + '/shap/' + discretization_type.lower() + '/shap_data_bip_mod_' + model + '.csv'
        global_code += r"\begin{minipage}{0.25\textwidth}" + "\n"
        global_code += summarize_attributes_importance(path, db_name, discretization_type, model)
        global_code += r"\end{minipage}" + "\n"
        if model != models1[-1]:
            global_code += r"\hspace{2cm}" + "\n"
    global_code += r"\end{figure}" + "\n"
    global_code += r"\vspace{-1cm}" + "\n"
    global_code += r"\begin{figure}[H]" + "\n"
    global_code += r"\hspace{-1cm}" + "\n"
    for model in models2:
        path = 'outputs/general_results/results/' + db_name + '/shap/' + discretization_type.lower() + '/shap_data_bip_mod_' + model + '.csv'
        global_code += r"\begin{minipage}{0.25\textwidth}" + "\n"
        global_code += summarize_attributes_importance(path, db_name, discretization_type, model)
        global_code += r"\end{minipage}" + "\n"
        if model != models2[-1]:
            global_code += r"\hspace{2cm}" + "\n"
    global_code += r"\caption{Features importance on " + db_name + " dataset case of " + discretization_type.lower() + " discretization}"+ "\n"
    global_code += r"\end{figure}" + "\n"

    directory = 'outputs/general_results/results/' + db_name + '/shap/'+discretization_type + '/summarize/'
    os.makedirs(directory, exist_ok=True)
    with open(directory+ 'attributes_importance.tex', 'w') as file:
        file.write(global_code)