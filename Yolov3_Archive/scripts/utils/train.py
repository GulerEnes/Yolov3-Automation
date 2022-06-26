import os


def train(cfgDict):
    print(f"config.cfg = {cfgDict}")
    runCommand = "python3 ../yolov3/train.py "

    for key, value in cfgDict.items():
        if key == 'model-path' or key == 'calculate_anchors' or key == 'create-data-file' or \
                key == 'anchor-custer-num' or key == 'anchor-width' or key == 'anchor-height':
            continue

        if str(value) == 'True':
            runCommand += ' --' + key
        elif str(value) == 'False':
            continue
        else:
            runCommand += ' --' + key + ' ' + value

    print('\n\n')
    print(f"ran command = {runCommand}")

    os.system(runCommand)


if __name__ == '__main__':
    pass
