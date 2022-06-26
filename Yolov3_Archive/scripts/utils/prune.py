import os
import utils.utils as uu


def runSystemCommand(command):
    os.system(command)


def duplicateFromeBase(baseModelPath, prunedModelsPath, prunePercentage):
    """Duplicating model to a new path and return it."""
    print(f"Duplicate from: {baseModelPath}")
    baseModelName = getModelName(baseModelPath)

    cmd = f"cp -r {baseModelPath} {prunedModelsPath}/{baseModelName}_pruned_{prunePercentage}"
    runSystemCommand(cmd)
    return f"{prunedModelsPath}/{baseModelName}_pruned_{prunePercentage}"


def creatingNotetxt(baseModelName, specificPrunedModelPath, prunePercentage):
    print("Creating note.txt for each Prune folder")
    notetxtPath = f"{specificPrunedModelPath}/note.txt"
    with open(notetxtPath, 'w') as notetxt:
        notetxt.write(f"Derived from {baseModelName} \n %{int(prunePercentage * 100)} pruning applied\n")


def runPruningScript(specificPrunedModelPath, percentage, imgSize):
    print("Run pruning script")
    cfgPath = uu.findModelCfgPath(specificPrunedModelPath)
    dataPath = f"{specificPrunedModelPath}/data.data"
    weightsPath = f"{specificPrunedModelPath}/best.pt"

    scriptPath = "../yolov3-channel-and-layer-pruning/prune.py"
    cmd = f"python3 {scriptPath} --cfg {cfgPath} --data {dataPath} --weights {weightsPath} --percent {percentage} --imgSize {imgSize}"
    runSystemCommand(cmd)


def testPrunedModelAndOutputIt(testDict, specificPrunedModelPath, prunedGraphDataPath, prunePercentage):
    print("Testing pruned model and write outputs to a .txt file")

    prunedWeights = f"{specificPrunedModelPath}/best.weights"
    prunedCfg = uu.findModelCfgPath(specificPrunedModelPath)
    runCommand = "python3 ../yolov3/test.py "
    for key, value in testDict.items():
        # Special cases
        if key == 'model-path' or key == 'save-logs':
            continue
        if key == 'weights':
            value = prunedWeights
        if key == 'cfg':
            value = prunedCfg

            # Appending flags to command
        if str(value) == 'True':
            runCommand += ' --' + key
        elif str(value) == 'False':
            continue
        else:
            runCommand += ' --' + key + ' ' + value

    runCommadOutputPath = f"{prunedGraphDataPath}/Prune_{prunePercentage}.txt"
    print(f"ran command = {runCommand}" + ' > ' + runCommadOutputPath)
    runSystemCommand(runCommand + ' > ' + runCommadOutputPath)

    print(f"DONE: {int(prunePercentage * 100)}/100")


def getModelName(baseModelPath):
    return baseModelPath.split('/')[-1]


def prune(testDict, pruneDict):
    start = int(pruneDict['prune-start'])
    finish = int(pruneDict['prune-finish'])
    step = int(pruneDict['prune-step'])
    baseModelPath = pruneDict['prune-base-model']
    imgSize = int(pruneDict['prune-img-size'])

    baseModelName = getModelName(baseModelPath)

    # Creating neccesary folders
    prunedModelsPath = f"../pruned_models/{baseModelName}/models"
    prunedGraphDataPath = f"../pruned_models/{baseModelName}/graph_data"

    runSystemCommand(f"mkdir ../pruned_models/{baseModelName}")
    runSystemCommand(f"mkdir {prunedModelsPath}")
    runSystemCommand(f"mkdir {prunedGraphDataPath}")

    for prunePercentage in range(start, finish, step):
        prunePercentage = round(prunePercentage / 100,
                                2)  # Converting percentage from int to float to make it between 0-1

        # Duplicate from base
        specificPrunedModelPath = duplicateFromeBase(baseModelPath, prunedModelsPath, prunePercentage)

        # Creating note.txt for each Prune folder
        creatingNotetxt(baseModelName, specificPrunedModelPath, prunePercentage)

        # Overwriting .data files to change paths
        uu.prepareData(specificPrunedModelPath)

        # Run pruning script
        runPruningScript(specificPrunedModelPath, prunePercentage, imgSize)

        # Converting .weights file to .pt
        uu.weights2pt(uu.findModelCfgPath(specificPrunedModelPath), f"{specificPrunedModelPath}/best.weights")

        # Testing pruned model and write outputs to a .txt file
        testPrunedModelAndOutputIt(testDict, specificPrunedModelPath, prunedGraphDataPath, prunePercentage)

    print("** DONE ** ")


if __name__ == '__main__':
    pass
