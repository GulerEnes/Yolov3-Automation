import sys

sys.path.append('../utils')

import utils as u
import glob
import os


def runSystemCommand(command):
    os.system(command)


if __name__ == '__main__':
    config = u.parseConfig("./config.cfg")

    mainPath = config['general']['main_data_folder_path']
    mainFolderName = mainPath.split('/')[-1]

    outPath = mainPath
    # outPath = mainPath.split(mainFolderName)[0] + f'prepared_{mainFolderName}'

    # print(f"Copying main folder {mainPath} to {outPath}   to work on it")
    # runSystemCommand(f'cp -r {mainPath} {outPath}')

    imagesFolders = glob.glob(f'{outPath}/**/images', recursive=True)
    labelsFolders = [i.replace('/images', '/labels') for i in imagesFolders]

    print(imagesFolders)
    print(labelsFolders)

    if config['general']['clean-empty-labels'] == True:
        for labelsFolder in labelsFolders:
            u.cleanEmptyLabels(labelsFolder)

    if config['general']['seperate-labels'] == True:
        for labelsFolder in labelsFolders:
            u.seperateLabels(labelsFolder)

    if config['general']['match-images-and-labels'] == True:
        for imagesFolder in imagesFolders:
            u.matchImagesAndLabels(imagesFolder)

    if config['general']['fix-label'] == True:
        for indx, labelsFolder in enumerate(labelsFolders):
            if int(config['fix-label']['new-class']) == -1:
                u.fixLabel(labelsFolder, indx)
            else:
                u.fixLabel(labelsFolder, config['fix-label']['new-class'])

    if config['general']['path-writer'] == True:
        txtPaths = []
        for imagesFolder in imagesFolders:
            txtPaths.append(u.pathWriter(imagesFolder))
        u.pathFilesCombiner(txtPaths, outPath + '/main_train.txt')

    if config['general']['vid-to-imgs'] == True:
        vids = glob.glob(f'{outPath}/**/*.avi', recursive=True)
        print(vids)
        for vid in vids:
            pieces = vid.split('/')
            videoname = pieces[-1].split('.')[0]
            outpath = '/'.join(pieces[:-1]) + '/' + videoname
            resize = config['vid-to-imgs']['resize'].split(',')

            u.vid2imgs(vid, outpath, config['vid-to-imgs']['out-img-format'],
                       -1, (int(resize[0]), int(resize[1])))

    if config['general']['vid-to-vids'] == True:
        vids = glob.glob(f'{outPath}/**/*.avi', recursive=True)
        print(vids)
        for vid in vids:
            pieces = vid.split('/')
            videoname = pieces[-1].split('.')[0]
            outpath = '/'.join(pieces[:-1]) + '/' + videoname + '_videos'
            resize = config['vid-to-vids']['resize'].split(',')

            u.vid2vids(vid, outpath, config['vid-to-vids']['out-vid-format'],
                       int(config['vid-to-vids']['seconds-per-folder']), (int(resize[0]), int(resize[1])))

    if config['general']['split-day-and-night'] == True:
        for imagesFolder in imagesFolders:
            u.dayNightSplitter(imagesFolder)

    if config['general']['data-augmentation'] == True:
        for imagesFolder in imagesFolders:
            u.dataAugmentation(imagesFolder, config['data-augmentation'])

    if config['general']['json-to-yolo'] == True:
        jsonFiles = glob.glob(f'{outPath}/**/*.json', recursive=True)
        u.json2yolo(jsonFiles)

    if config['general']['yolo-to-json'] == True:
        # Folders must have images and labels folders. Also must have obj.names file.
        for labelFolder in labelsFolders:
            generalFolder = labelFolder.split('/labels')[0]
            u.yolo2json(generalFolder, generalFolder + '/labels.json', generalFolder + '/obj.names')

    if config['general']['yolo-to-xml'] == True:
        # Folders must have images and labels folders.
        for labelFolder in labelsFolders:
            generalFolder = labelFolder.split('/labels')[0]
            labelNames = list(map(str, config['yolo-to-xml']['labels'].split(',')))

            u.yolo2xml(generalFolder, labelNames)
