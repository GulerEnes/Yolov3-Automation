import os
import utils.utils as uu


def runSystemCommand(command):
    os.system(command)


def testRunCommand(testDict, imgSize, modelPath):
    runCommand = "python3 ../yolov3/test.py "
    modelWeights = f"{modelPath}/best.pt"
    modelCfg = uu.findModelCfgPath(modelPath)
    for key, value in testDict.items():
        if key == 'model-path' or key == 'save-logs':
            continue
        if key == 'weights':
            value = modelWeights
        if key == 'cfg':
            value = modelCfg
        if key == 'img-size':
            value = str(imgSize)
        if str(value) == 'True':
            runCommand += ' --' + key
        elif str(value) == 'False':
            continue
        else:
            runCommand += ' --' + key + ' ' + value

    return runCommand


def imgSizeTest(imgSizeTestDict, testDict):
    step = int(imgSizeTestDict['img-size-test-step'])
    minn = int(imgSizeTestDict['img-size-test-min'])
    maxx = int(imgSizeTestDict['img-size-test-max'])
    baseModelPath = imgSizeTestDict['img-size-test-base-model']
    base = baseModelPath.split('/')[-1]

    # Creating a folder to store output txt's 
    runSystemCommand(f"mkdir ../img_size_tests/img_size_graph_data_{base}_{minn}_{maxx}_{step}")

    for imgSize in range(minn, maxx + 1, step):
        # Testing model and write outputs to a .txt file
        print("Testing model and writing outputs to a .txt file")
        cmd = f"{testRunCommand(testDict, imgSize, baseModelPath)} > ../img_size_tests/img_size_graph_data_{base}_{minn}_{maxx}_{step}/img_size_{imgSize}.txt"
        runSystemCommand(cmd)

        print(f"DONE: {imgSize}/{maxx}  step: {step}")

    print("** DONE ** ")


if __name__ == '__main__':
    pass
