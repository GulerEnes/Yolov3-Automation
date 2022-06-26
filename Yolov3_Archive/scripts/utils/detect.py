import os


def detect(cfgDict):
    print(f"config.cfg = {cfgDict}")
    runCommand = "python3 ../yolov3/detect.py "

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
    print(f"ran command = {runCommand}")

    modelName = cfgDict['model-path'].split('/')[-1]
    imgSize = cfgDict['img-size']
    if str(cfgDict['save-logs']) == 'True':
        os.system(runCommand + f" > ../model_detect_outputs/{modelName}/log_{imgSize}.txt")
    else:
        os.system(runCommand)


if __name__ == '__main__':
    pass
