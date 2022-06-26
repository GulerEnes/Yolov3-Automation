import os


def test(cfgDict):
    print(f"config.cfg = {cfgDict}")
    runCommand = "python3 ../yolov3/test.py "

    for key, value in cfgDict.items():
        if key == 'model-path' or key == 'save-logs':
            continue

        if str(value) == 'True':
            runCommand += ' --' + key
        elif str(value) == 'False':
            continue
        else:
            runCommand += ' --' + key + ' ' + value

    print('\n\n')

    modelName = cfgDict['model-path'].split('/')[-1]
    imgSize = cfgDict['img-size']
    if str(cfgDict['save-logs']) == 'True':
        logFilePath = f"../model_test_outputs/{modelName}_log_{imgSize}.txt"
        print(f"ran command = {runCommand}" + f" > {logFilePath}")
        os.system(f"{runCommand} > {logFilePath}")
    else:
        print(f"ran command = {runCommand}")
        os.system(runCommand)


if __name__ == '__main__':
    pass
