from os.path import splitext
from os import walk
import matplotlib.pyplot as plt
import os
import numpy as np


def getValues(path):
    with open(path, 'r') as f:
        lines = f.readlines()

        valsLine = ""
        for line in lines:
            if "all " in line:
                valsLine = line
                break
        if valsLine == '':
            return {'P': 0, 'R': 0, 'mAP': 0, 'F1': 0}
        #                   all   1.2e+03   1.2e+03     0.983     0.984     0.992     0.984
        vals = [i for i in valsLine.split(' ') if i != '']
        precision = float(vals[3])
        recall = float(vals[4])
        mAP = float(vals[5])
        f1 = float(vals[6].split('\n')[0])

        return {'P': precision, 'R': recall, 'mAP': mAP, 'F1': f1}


def allValsForABase(path):
    values = []

    try:
        _, _, filenames = next(walk(path))

        for f in filenames:
            filename, ext = splitext(f)
            if ext == '.txt':
                filepath = path + '/' + f

                prune = int(filename.split('_')[-1])

                vals = getValues(filepath)

                values.append((prune, vals))

    except Exception as e:
        print("Error:", e)

    values = list(filter(None, values))
    return values


def imgSizePlotGraph(imgSizeTestDict):
    step = int(imgSizeTestDict['img-size-test-step'])
    minn = int(imgSizeTestDict['img-size-test-min'])
    maxx = int(imgSizeTestDict['img-size-test-max'])

    baseModels = list(imgSizeTestDict['img-size-test-graph-base-models'].split(' '))

    print(baseModels)
    modelPruneVals = list()

    for base in baseModels:
        path = f"../img_size_tests/img_size_graph_data_{base}_{minn}_{maxx}_{step}"

        modelPruneVals.append([base, allValsForABase(path)])

    for i in range(len(modelPruneVals)):
        modelPruneVals[i][1] = sorted(modelPruneVals[i][1], key=lambda x: x[0])

    X = [prune for prune, v in modelPruneVals[0][1]]

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2, figsize=(25, 25))

    x = np.arange(minn, maxx + 1, step)

    # Precision
    axis[0, 0].set_title("Precision")
    for base, vals in modelPruneVals:
        precisionY = [v["P"] for prune, v in vals]
        axis[0, 0].plot(X, precisionY, label=base, linewidth=1)
    axis[0, 0].legend()
    axis[0, 0].grid()
    axis[0, 0].set_xticks(x)
    axis[0, 0].tick_params(axis='x', rotation=90)

    # Recall
    axis[0, 1].set_title("Recall")
    for base, vals in modelPruneVals:
        precisionY = [v["R"] for prune, v in vals]
        axis[0, 1].plot(X, precisionY, label=base, linewidth=1)
    axis[0, 1].legend()
    axis[0, 1].grid()
    axis[0, 1].set_xticks(x)
    axis[0, 1].tick_params(axis='x', rotation=90)

    # mAP
    axis[1, 0].set_title("mAP")
    for base, vals in modelPruneVals:
        precisionY = [v["mAP"] for prune, v in vals]
        axis[1, 0].plot(X, precisionY, label=base, linewidth=1)
    axis[1, 0].legend()
    axis[1, 0].grid()
    axis[1, 0].set_xticks(x)
    axis[1, 0].tick_params(axis='x', rotation=90)

    # F1
    axis[1, 1].set_title("F1")
    for base, vals in modelPruneVals:
        precisionY = [v["F1"] for prune, v in vals]
        axis[1, 1].plot(X, precisionY, label=base, linewidth=1)
    axis[1, 1].legend()
    axis[1, 1].grid()
    axis[1, 1].set_xticks(x)
    axis[1, 1].tick_params(axis='x', rotation=90)

    # Combine all the operations and display
    plt.savefig(f"../img_size_tests/img_size_prmf_{'_'.join(baseModels)}_{minn}_{maxx}_{step}.png")
