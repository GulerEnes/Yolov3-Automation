import os
import sys

sys.path.append('../Dataset_Preparation/')
sys.path.append('./utils/')
import glob
import vid2imgs as v2i
import vid2vids as v2v
import cv2 as cv
from tqdm import tqdm
import shutil
import numpy as np
from random import randint
import skimage as skimage
# from json_parser import Parser
import json
from lxml import etree
from PIL import Image
from pathlib import Path
import csv


# This class prepared for labelled data for json file.
# You can use, if you want to json parser and convert whatever you want format
class Parser(object):
    def __init__(self):
        print("object is created!")

    def parse(self, path):
        raw = {}
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            parsedData = self.process(raw)
            return parsedData

        except (IOError, AttributeError) as e:
            print(str(e))
            return None

    def process(self, raw):
        pass

    # class VggObjExistParser(Parser):
    def process(self, raw):

        data = raw['_via_img_metadata']
        tablu = {}
        for key, value in data.items():
            print(key)
            tablu[value["filename"]] = value['regions']
        data = tablu
        cleanedData = {}
        for key, value in data.items():
            if len(value) > 0:
                cleanedData[key] = value

        polished = {}
        for key, value in cleanedData.items():
            arr = []
            for el in value:
                el = el['shape_attributes']
                arr.append({'x': el['x'], 'y': el['y'], 'w': el['width'], 'h': el['height']})
            polished[key] = arr
        # print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in polished.items()) + "}")
        return polished


def deleteComment(line):
    try:
        commentStartInd = line.index('#')
        return line[:commentStartInd]
    except:  # substring not found
        return line


def parseConfig(configPath):
    configDict = dict()
    with open(configPath, 'r', encoding="utf-8") as cfgFile:
        # lines = cfgFile.read().split('\n')
        lines = cfgFile.readlines()

        currentBlock = ''
        for line in lines:
            line = deleteComment(line)  # deleting comments
            line = line.strip()  # deleting white spaces at beginnig and ending of line
            if line == '':
                continue

            if line[0] == '[':  # New block starts
                configDict[line[1:-1]] = dict()
                currentBlock = line[1:-1]
                continue

            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if value == 'True':
                value = True
            if value == 'False':
                value = False

            configDict[currentBlock][key] = value

    return configDict


def findModelCfgPath(modelPath):
    _, _, filenames = next(os.walk(modelPath))
    for file in filenames:
        if file[-4:] == '.cfg':
            return os.path.join(modelPath, file)


def countClassesOfObjnames(objNamesPath):
    count = 0
    with open(objNamesPath, 'r', encoding="utf-8") as objNamesFile:
        lines = objNamesFile.readlines()

        for line in lines:
            if line.strip() != '':
                count += 1
    return count


def prepareData(modelPath):
    dataPath = os.path.join(modelPath, 'data.data')
    objNamesPath = os.path.join(modelPath, 'obj.names')
    trainPath = os.path.join(modelPath, 'train.txt')
    validPath = os.path.join(modelPath, 'validate.txt')

    with open(dataPath, 'w', encoding="utf-8") as dataFile:
        numOfClasses = countClassesOfObjnames(objNamesPath)
        dataFile.write(f"classes={numOfClasses}\n")
        dataFile.write(f"train={trainPath}\n")
        dataFile.write(f"valid={validPath}\n")
        dataFile.write(f"names={objNamesPath}\n")


def pt2weights(cfgPath, ptPath="../yolov3/weights/best.pt"):
    command = f"python3 -c \"import sys; sys.path.append('../yolov3/'); from models import *; convert('{cfgPath}', '{ptPath}')\""
    os.system(command)


def weights2pt(cfgPath, weightPath="../yolov3/weights/best.weight"):
    command = f"python3  -c \"import sys; sys.path.append('../yolov3/'); from models import *; convert('{cfgPath}', '{weightPath}')\""
    os.system(command)


def moveWeightsToModelPath(modelPath, weightPath="../yolov3/weights"):
    os.system(f"mv {weightPath}/best.pt {modelPath}")
    os.system(f"mv {weightPath}/last.pt {modelPath}")
    os.system(f"mv {weightPath}/best.weights {modelPath}")


def moveResultsToModelPath(modelPath, resultsPath="./"):
    os.system(f"mv {resultsPath}/results.png {modelPath}")
    os.system(f"mv {resultsPath}/results.txt {modelPath}")


def mkdirDetectOutputFolder(path, force=False):
    if not force and os.path.isdir(path):
        raise "Output path is already exist! To not overwride it program stopping."
    os.system(f"mkdir {path}")


def changeAnchorsInModelCfg(numCluster, modelCfgPath):
    ancFile = open(f'generated_anchors/anchors{numCluster}.txt', 'r', encoding="utf-8")
    newAnc = ancFile.readlines()[0]
    newAnc = ",".join([str(round(float(i) * 32)) for i in newAnc.split(',')])
    ancFile.close()

    modelCfgFile = open(modelCfgPath, 'r', encoding="utf-8")
    lines = modelCfgFile.readlines()
    modelCfgFile.close()

    modelCfgFile = open(modelCfgPath, 'w', encoding="utf-8")
    for line in lines:
        if 'anchors' in line:
            modelCfgFile.write('anchors = ' + newAnc + '\n')
        else:
            modelCfgFile.write(line)
    modelCfgFile.close()


def cleanEmptyLabels(labelsPath):
    """
    This function removes txt's if they 
    are totally empty,
    has only \n's,
    has only \n's and white spaces.
    """
    labelTxts = glob.glob(f'{labelsPath}/*.txt')

    print('Starting to clean empty labels under ', labelsPath)
    characterFoundFlag = False
    for indx, labelTxt in enumerate(labelTxts):
        print(f'\r{indx}/{len(labelTxts)}', end='       ')
        with open(labelTxt, 'r') as f:
            lines = [i.replace('\n', '') for i in f.readlines()]
            lines = [i.replace(' ', '') for i in lines]
            for line in lines:
                if len(line) > 1:
                    characterFoundFlag = True
                    break
        if not characterFoundFlag:
            os.remove(labelTxt)
            print(f"\nREMOVING: {labelTxt}")
        characterFoundFlag = False
    print("\n** Done ** ", labelsPath)


def seperateLabels(labelsPath):
    seperatedFolderPath = labelsPath.replace('/labels', '/seperated')
    os.mkdir(seperatedFolderPath)
    labelTxts = glob.glob(f'{labelsPath}/*.txt')
    print('Starting to seperate labels under ', labelsPath)
    for indx, labelTxt in enumerate(labelTxts):
        print(f'\r{indx}/{len(labelTxts)}', end='       ')
        imagePath = labelTxt.replace('/labels', '/images')
        imagePath = imagePath.replace('txt', 'jpg')
        with open(labelTxt, 'r') as f:
            lines = f.readlines()
            classPath = ""

            for line in lines:
                line = line.replace('\n', '')
                classNumber = line.split(' ')[0]
                classPath = f"{seperatedFolderPath}/{classNumber}"
                if not os.path.exists(classPath):
                    os.mkdir(classPath)
                    os.mkdir(classPath + '/images')
                    os.mkdir(classPath + '/labels')
            os.system(f"cp {imagePath} {classPath}/images/")
            os.system(f"cp {labelTxt} {classPath}/labels/")
    print("\n** Done ** ", labelsPath)


def matchImagesAndLabels(imagesFilePath):
    labelsPath = imagesFilePath.replace('/images', '/labels')

    labelTxts = set(glob.glob(f'{labelsPath}/*.txt'))
    images = glob.glob(f'{imagesFilePath}/*.jpg')

    print('Starting to match images with labels under ', imagesFilePath)
    for indx, image in enumerate(images):
        print(f'\r{indx}/{len(labelTxts)}', end='       ')
        label = image.replace('.jpg', '.txt')
        label = label.replace('/images', '/labels')
        if label not in labelTxts:
            print(f"\nRemoving {image}, because it has not a label")
            os.remove(image)
    labelTxts = set(glob.glob(f'{labelsPath}/*.txt'))
    images = glob.glob(f'{imagesFilePath}/*.jpg')
    for indx, label in enumerate(labelTxts):
        print(f'\r{indx}/{len(labelTxts)}', end='       ')
        image = label.replace('.txt', '.jpg')
        image = image.replace('/labels', '/images')
        if image not in images:
            print(f"\nRemoving {label}, because it has not a label")
            os.remove(label)
    print("\n** Done ** ", imagesFilePath)


def fixLabel(labelsPath, newClass):
    labelTxts = glob.glob(f'{labelsPath}/*.txt')
    print('Starting to fix', labelsPath)
    for indx, labelTxt in enumerate(labelTxts):
        fr = open(labelTxt, 'r')
        lines = fr.readlines()
        fr.close()

        with open(labelTxt, 'w') as fw:
            for line in lines:
                if len(line) > 5:
                    fw.write(str(newClass) + ' ' + ' '.join(line.split(' ')[1:]))
        print(f'\r{indx}/{len(labelTxts)}', end='       ')
    print("\n** Done ** ", labelsPath)


def pathWriter(imagesPath):
    images = glob.glob(f'{imagesPath}/*.jpg')
    traintxtPath = imagesPath.replace('/images', '/train.txt')
    print('Starting to write image paths under ', imagesPath)

    with open(traintxtPath, 'w') as f:
        for indx, image in enumerate(images):
            print(f'\r{indx}/{len(images)}', end='       ')
            f.write(image + '\n')
    print("\n** Done ** ", imagesPath)
    return traintxtPath


def pathFilesCombiner(txtPaths, mainTxtPath):
    with open(mainTxtPath, 'w') as mf:
        for txtPath in txtPaths:
            with open(txtPath, 'r') as f:
                lines = f.readlines()
                mf.writelines(lines)
    print("Main train txt file is written to ", mainTxtPath)


def vid2imgs(inpVideoPath, outFolderPath, outFormat, impPerFolder, resize):
    v2i.main(inpVideoPath, outFolderPath, outFormat, impPerFolder, resize)


def vid2vids(inpVideoPath, outFolderPath, outFormat, seconds, resize):
    v2v.main(inpVideoPath, outFolderPath, outFormat, int(seconds), resize)


def dayNightSplitter(imagesFolderPath):
    print('Starting to split day-night images under ', imagesFolderPath)
    dayPath = imagesFolderPath.replace('/images', '/day')
    nightPath = imagesFolderPath.replace('/images', '/night')

    os.mkdir(dayPath)
    os.mkdir(nightPath)

    os.mkdir(f"{dayPath}/images")
    os.mkdir(f"{dayPath}/labels")
    os.mkdir(f"{nightPath}/images")
    os.mkdir(f"{nightPath}/labels")

    imagePaths = glob.glob(f"{imagesFolderPath}/*.jpg")

    for imagePath in tqdm(imagePaths):
        processHSL(imagePath, dayPath, nightPath)


def processHSL(imagePath, dayPath, nightPath):
    labelPath = imagePath.replace('/images', '/labels')
    labelPath = labelPath.replace('.jpg', '.txt')

    img = cv.imread(imagePath)
    imgHsl = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    light = imgHsl[:, :, 1]
    satu = imgHsl[:, :, 2]

    if light.mean() < 120 and satu.mean() < 18:  # NIGHT
        shutil.copy(imagePath, f"{nightPath}/images")
        shutil.copy(labelPath, f"{nightPath}/labels")
    else:  # DAY
        shutil.copy(imagePath, f"{dayPath}/images")
        shutil.copy(labelPath, f"{dayPath}/labels")


def dataAugmentation(imagesFolder, optionsDict):
    augmentedPath = createFoldersForDataAugmentation(imagesFolder, optionsDict)
    imagePaths = glob.glob(f"{imagesFolder}/*.jpg")

    for imagePath in tqdm(imagePaths):
        labelPath = imagePath.replace('/images', '/labels')
        labelPath = labelPath.replace('.jpg', '.txt')

        imgName = imagePath.split('/')[-1]
        labelName = labelPath.split('/')[-1]
        img = cv.imread(imagePath)
        out = None
        if optionsDict['brightness'] != '':
            if optionsDict['brightness'] == 'random':
                out = brightness(img)
            else:
                out = brightness(img, optionsDict['brightness'])
            cv.imwrite(f'{augmentedPath}/brightness/images/brightness_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/brightness/labels/brightness_{labelName}')

        if optionsDict['saturation'] != '':
            if optionsDict['saturation'] == 'random':
                out = saturation(img)
            else:
                out = saturation(img, optionsDict['saturation'])
            cv.imwrite(f'{augmentedPath}/saturation/images/saturation_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/saturation/labels/saturation_{labelName}')

        if optionsDict['hue'] != '':
            if optionsDict['hue'] == 'random':
                out = hue(img)
            else:
                out = hue(img, optionsDict['hue'])
            cv.imwrite(f'{augmentedPath}/hue/images/hue_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/hue/labels/hue_{labelName}')

        if optionsDict['blur'] != '':
            kernel = tuple(map(int, optionsDict['blur'].split(',')))
            out = blurImage(img, kernel)
            cv.imwrite(f'{augmentedPath}/blur/images/blur_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/blur/labels/blur_{labelName}')

        if optionsDict['median-blur'] != '':
            out = medianImage(img, optionsDict['median-blur'])
            cv.imwrite(f'{augmentedPath}/median_blur/images/median_blur_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/median_blur/labels/median_blur_{labelName}')

        if optionsDict['filter2d'] != '':
            kernel = tuple(map(int, optionsDict['filter2d'].split(',')))
            out = filter2d(img, kernel)
            cv.imwrite(f'{augmentedPath}/filter2d/images/filter2d_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/filter2d/labels/filter2d_{labelName}')

        if optionsDict['convert-gray'] == True:
            out = grayScale(img)
            cv.imwrite(f'{augmentedPath}/gray/images/gray_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/gray/labels/gray_{labelName}')

        if optionsDict['random-crop'] == True:
            out = randomCrop(img)
            cv.imwrite(f'{augmentedPath}/random_crop/images/random_crop_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/random_crop/labels/random_crop_{labelName}')
        if optionsDict['random-erasing'] == True:
            out = randomErasing(img)
            cv.imwrite(f'{augmentedPath}/random_erasing/images/random_erasing_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/random_erasing/labels/random_erasing_{labelName}')
        if optionsDict['random-contrast'] == True:
            out = randomContrast(img)
            cv.imwrite(f'{augmentedPath}/random_contrast/images/random_contrast_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/random_contrast/labels/random_contrast_{labelName}')
        if optionsDict['random-noise'] == True:
            out = noise(img)
            cv.imwrite(f'{augmentedPath}/random_noise/images/random_noise_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/random_noise/labels/random_noise_{labelName}')
        if optionsDict['random-rotation'] == True:
            out = randomRotation(img)
            cv.imwrite(f'{augmentedPath}/random_rotation/images/random_rotation_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/random_rotation/labels/random_rotation_{labelName}')
        if optionsDict['flip-horizontal'] == True:
            out = flipHorizontal(img)
            cv.imwrite(f'{augmentedPath}/flip_horizontal/images/flip_horizontal_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/flip_horizontal/labels/flip_horizontal_{labelName}')
        if optionsDict['flip-vertical'] == True:
            out = flipVertical(img)
            cv.imwrite(f'{augmentedPath}/flip_vertical/images/flip_vertical_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/flip_vertical/labels/flip_vertical{labelName}')
        if optionsDict['flip-origin'] == True:
            out = flipOrigin(img)
            cv.imwrite(f'{augmentedPath}/flip_origin/images/flip_origin_{imgName}', out)
            shutil.copy(labelPath, f'{augmentedPath}/flip_origin/labels/flip_origin_{labelName}')


def createFoldersForDataAugmentation(imagesFolder, optionsDict):
    augmentedPath = imagesFolder.replace('/images', '/augmented')
    os.mkdir(augmentedPath)

    if optionsDict['brightness'] != '':
        os.mkdir(f"{augmentedPath}/brightness")
        os.mkdir(f"{augmentedPath}/brightness/images")
        os.mkdir(f"{augmentedPath}/brightness/labels")
    if optionsDict['saturation'] != '':
        os.mkdir(f"{augmentedPath}/saturation")
        os.mkdir(f"{augmentedPath}/saturation/images")
        os.mkdir(f"{augmentedPath}/saturation/labels")
    if optionsDict['hue'] != '':
        os.mkdir(f"{augmentedPath}/hue")
        os.mkdir(f"{augmentedPath}/hue/images")
        os.mkdir(f"{augmentedPath}/hue/labels")
    if optionsDict['blur'] != '':
        os.mkdir(f"{augmentedPath}/blur")
        os.mkdir(f"{augmentedPath}/blur/images")
        os.mkdir(f"{augmentedPath}/blur/labels")
    if optionsDict['median-blur'] != '':
        os.mkdir(f"{augmentedPath}/median_blur")
        os.mkdir(f"{augmentedPath}/median_blur/images")
        os.mkdir(f"{augmentedPath}/median_blur/labels")
    if optionsDict['filter2d'] != '':
        os.mkdir(f"{augmentedPath}/filter2d")
        os.mkdir(f"{augmentedPath}/filter2d/images")
        os.mkdir(f"{augmentedPath}/filter2d/labels")
    if optionsDict['convert-gray'] == True:
        os.mkdir(f"{augmentedPath}/gray")
        os.mkdir(f"{augmentedPath}/gray/images")
        os.mkdir(f"{augmentedPath}/gray/labels")
    if optionsDict['random-crop'] == True:
        os.mkdir(f"{augmentedPath}/random_crop")
        os.mkdir(f"{augmentedPath}/random_crop/images")
        os.mkdir(f"{augmentedPath}/random_crop/labels")
    if optionsDict['random-erasing'] == True:
        os.mkdir(f"{augmentedPath}/random_erasing")
        os.mkdir(f"{augmentedPath}/random_erasing/images")
        os.mkdir(f"{augmentedPath}/random_erasing/labels")
    if optionsDict['random-contrast'] == True:
        os.mkdir(f"{augmentedPath}/random_contrast")
        os.mkdir(f"{augmentedPath}/random_contrast/images")
        os.mkdir(f"{augmentedPath}/random_contrast/labels")
    if optionsDict['random-noise'] == True:
        os.mkdir(f"{augmentedPath}/random_noise")
        os.mkdir(f"{augmentedPath}/random_noise/images")
        os.mkdir(f"{augmentedPath}/random_noise/labels")
    if optionsDict['random-rotation'] == True:
        os.mkdir(f"{augmentedPath}/random_rotation")
        os.mkdir(f"{augmentedPath}/random_rotation/images")
        os.mkdir(f"{augmentedPath}/random_rotation/labels")
    if optionsDict['flip-horizontal'] == True:
        os.mkdir(f"{augmentedPath}/flip_horizontal")
        os.mkdir(f"{augmentedPath}/flip_horizontal/images")
        os.mkdir(f"{augmentedPath}/flip_horizontal/labels")
    if optionsDict['flip-vertical'] == True:
        os.mkdir(f"{augmentedPath}/flip_vertical")
        os.mkdir(f"{augmentedPath}/flip_vertical/images")
        os.mkdir(f"{augmentedPath}/flip_vertical/labels")
    if optionsDict['flip-origin'] == True:
        os.mkdir(f"{augmentedPath}/flip_origin")
        os.mkdir(f"{augmentedPath}/flip_origin/images")
        os.mkdir(f"{augmentedPath}/flip_origin/labels")
    return augmentedPath


def brightness(img, value='random'):
    if value == 'random':
        value = randint(-100, 100)
    value = int(value)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    finalHsv = cv.merge((h, s, v))
    img = cv.cvtColor(finalHsv, cv.COLOR_HSV2BGR)
    return img


def saturation(img, value='random'):
    if value == 'random':
        value = randint(-100, 100)
    value = int(value)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    s = cv.add(s, value)
    s[s > 255] = 255
    s[s < 0] = 0
    finalHsv = cv.merge((h, s, v))
    img = cv.cvtColor(finalHsv, cv.COLOR_HSV2BGR)
    return img


def hue(img, value='random'):
    if value == 'random':
        value = randint(-100, 100)
    value = int(value)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = cv.add(h, value)
    h[h > 255] = 255
    h[h < 0] = 0
    finalHsv = cv.merge((h, s, v))
    img = cv.cvtColor(finalHsv, cv.COLOR_HSV2BGR)
    return img


def blurImage(img, kernel=(7, 7)):
    blur = cv.blur(img, kernel)
    return blur


def medianImage(img, kernel=7):
    kernel = int(kernel)
    median = cv.medianBlur(img, kernel)
    return median


def filter2d(img, kernel=np.ones((5, 5), np.uint8) / 25):
    f2d = cv.filter2D(img, -1, kernel)
    return f2d


def grayScale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def randomCrop(img):
    width, height, chanel = img.shape
    x1 = randint(0, int(width * 0.7))
    y1 = randint(0, int(height * 0.7))

    edgeSize = randint(10, width)
    x2 = min(width - 1, x1 + edgeSize)
    y2 = min(height - 1, y1 + edgeSize)
    temp = img[x1:x2, y1:y2, :]
    return cv.resize(temp, (height, width))


def randomErasing(img):
    temp = img.copy()
    width, height, chanel = temp.shape
    x1 = randint(0, width)
    y1 = randint(0, height)

    x2 = randint(x1, width)
    y2 = randint(y1, height)

    temp[x1:x2, y1:y2, :] = 0
    return temp


def randomContrast(img):
    # alpha value [1.0-3.0], beta value [0-100]
    return cv.convertScaleAbs(img, alpha=randint(100, 300) / 100, beta=randint(0, 100))


def noise(img):
    out = skimage.util.random_noise(img, mode="gaussian")
    normImage = cv.normalize(out, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return normImage.astype(np.uint8)


def randomRotation(img):
    angle = randint(30, 330)
    height, width = img.shape[:2]
    rotateMatrix = cv.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=1)
    return cv.warpAffine(src=img, M=rotateMatrix, dsize=(width, height))


def flipHorizontal(img):
    return cv.flip(img, 1)


def flipVertical(img):
    return cv.flip(img, 0)


def flipOrigin(img):
    return cv.flip(img, -1)


def getImageSize(imgFile):
    img = cv.imread(imgFile)
    h = img.shape[0]
    w = img.shape[1]
    return h, w


def yoloFormat(parser, yoloLabelFolder):
    for key, value in parser.items():
        f = open(yoloLabelFolder + str(key)[:-4] + ".txt", "w+")
        for i in range(0, len(parser[key])):
            # print(parser[k][i])
            # These are not width and height but I don't know what they are. vgg16 annotation tool json data format xmin, ymin, xmax, ymax -merve
            xmin = parser[key][i].get("w")
            ymin = parser[key][i].get("h")
            xmax = parser[key][i].get("x") + (xmin / 2)
            ymax = parser[key][i].get("y") + (ymin / 2)
            h, w = getImageSize()
            f.write('{:d} '.format(0))  # class id
            f.write('{:f} '.format(xmax / w))
            f.write('{:f} '.format(ymax / h))
            f.write('{:f} '.format(xmin / w))
            f.write('{:f} \n'.format(ymin / h))


def createImageAnnotation(fileName, width, height, imageId):
    fileName = fileName.split('/')[-1]
    images = {
        'fileName': fileName,
        'height': height,
        'width': width,
        'id': imageId
    }
    return images


def createAnnotationYoloFormat(minX, minY, width, height, imageId, categoryId, annotationId):
    bbox = (minX, minY, width, height)
    area = width * height

    annotation = {
        'id': annotationId,
        'image_id': imageId,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': categoryId,
        'segmentation': []
    }

    return annotation


# Create the annotations of the ECP dataset (Coco format)
cocoFormat = {
    "images": [
        {
        }
    ],
    "categories": [

    ],
    "annotations": [
        {
        }
    ]
}


# Get 'images' and 'annotations' info
def imagesAnnotationsInfo(generalFolder):
    path = generalFolder + '/train.txt'
    # path : train.txt or test.txt
    annotations = []
    images = []

    file = open(path, "r")
    readLines = file.readlines()
    file.close()

    imageId = 0
    annotationId = 1  # In COCO dataset format, you must start annotation id with '1'

    for line in readLines:
        # Check how many items have progressed
        if imageId % 100 == 0:
            print("Processing " + str(imageId) + " ...")

        line = line.replace('\n', '')
        imgFile = cv.imread(line)

        # read a label file
        modLine = line.replace('images', 'labels')
        labelPath = modLine[:-3] + "txt"
        labelFile = open(labelPath, "r")
        labelReadLine = labelFile.readlines()
        labelFile.close()

        h, w, _ = imgFile.shape

        # Create image annotation
        image = createImageAnnotation(line, w, h, imageId)
        images.append(image)

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in labelReadLine:
            labelLine = line1
            categoryId = int(labelLine.split()[0]) + 1  # you start with annotation id with '1'
            xCenter = float(labelLine.split()[1])
            yCenter = float(labelLine.split()[2])
            width = float(labelLine.split()[3])
            height = float(labelLine.split()[4])

            intXCenter = int(imgFile.shape[1] * xCenter)
            intYCenter = int(imgFile.shape[0] * yCenter)
            intWidth = int(imgFile.shape[1] * width)
            intHeight = int(imgFile.shape[0] * height)

            min_x = intXCenter - intWidth / 2
            min_y = intYCenter - intHeight / 2
            width = intWidth
            height = intHeight

            annotation = createAnnotationYoloFormat(min_x, min_y, width, height, imageId, categoryId, annotationId)
            annotations.append(annotation)
            annotationId += 1

        imageId += 1  # if you finished annotation work, updates the image id.

    return images, annotations


def json2yolo(jsonFiles):
    for jsonFile in jsonFiles:
        json = Parser()
        parser = json.parse(jsonFile)

        # Creating yolo_label folder. If already exist nothing to do.
        yoloLabelFolder = "/".join(list(jsonFile.split('/')[:-2])) + '/yolo_labels/'
        if not os.path.exists(yoloLabelFolder):
            os.mkdir(yoloLabelFolder)

        yoloFormat(parser, yoloLabelFolder)
        print("finished")


def yolo2json(generalFolder, outputPath, objPath):
    classes = []
    with open(objPath) as f:
        for line in f:
            classes.append(line.split('\n')[0])
    print("Found Classes\n", classes)

    print("Start Process!")

    cocoFormat['images'], cocoFormat['annotations'] = imagesAnnotationsInfo(generalFolder)

    for index, label in enumerate(classes):
        ann = {
            "supercategory": "issd.com.tr/en",
            "id": index + 1,  # Index starts with '1' .
            "name": label
        }
        cocoFormat['categories'].append(ann)

    with open(outputPath, 'w') as outfile:
        json.dump(cocoFormat, outfile)
        print("Finished!")


global labels
labels = []

global label
label = ''


def csvread(fn):
    with open(fn, 'r') as csvfile:
        listArr = []
        reader = csv.reader(csvfile, delimiter=' ')

        for row in reader:
            listArr.append(row)
    return listArr


def convertLabel(txtFile):
    global label
    for i in range(len(labels)):
        if txtFile[0] == str(i):
            label = labels[i]
            return label

    return label


# core code = convert the yolo txt file to the x_min,x_max...


def extractCoor(txtFile, imgWidth, imgHeight):
    x_rect_mid = float(txtFile[1])
    y_rect_mid = float(txtFile[2])
    width_rect = float(txtFile[3])
    height_rect = float(txtFile[4])

    x_min_rect = ((2 * x_rect_mid * imgWidth) - (width_rect * imgWidth)) / 2
    x_max_rect = ((2 * x_rect_mid * imgWidth) + (width_rect * imgWidth)) / 2
    y_min_rect = ((2 * y_rect_mid * imgHeight) -
                  (height_rect * imgHeight)) / 2
    y_max_rect = ((2 * y_rect_mid * imgHeight) +
                  (height_rect * imgHeight)) / 2

    return x_min_rect, x_max_rect, y_min_rect, y_max_rect


def yolo2xmlmain(IMG_PATH, fw, txtFolder, savePath):
    for line in fw:
        root = etree.Element("annotation")

        # try debug to check your path
        imgStyle = IMG_PATH.split('/')[-1]
        imgName = line
        imageInfo = IMG_PATH + "/" + line
        imgTxtRoot = txtFolder + "/" + line[:-4]
        # print(imgTxtRoot)
        txt = ".txt"

        txtPath = imgTxtRoot + txt
        # print(txtPath)
        txtFile = csvread(txtPath)
        ######################################

        # read the image  information
        imgSize = Image.open(imageInfo).size

        imgWidth = imgSize[0]
        imgHeight = imgSize[1]
        imgDepth = Image.open(imageInfo).layers
        ######################################

        folder = etree.Element("folder")
        folder.text = "%s" % (imgStyle)

        filename = etree.Element("filename")
        filename.text = "%s" % (imgName)

        path = etree.Element("path")
        path.text = "%s" % (IMG_PATH)

        source = etree.Element("source")
        ##################source - element##################
        sourceDatabase = etree.SubElement(source, "database")
        sourceDatabase.text = "Unknown"
        ####################################################

        size = etree.Element("size")
        ####################size - element##################
        image_width = etree.SubElement(size, "width")
        image_width.text = "%d" % (imgWidth)

        image_height = etree.SubElement(size, "height")
        image_height.text = "%d" % (imgHeight)

        image_depth = etree.SubElement(size, "depth")
        image_depth.text = "%d" % (imgDepth)
        ####################################################

        segmented = etree.Element("segmented")
        segmented.text = "0"

        root.append(folder)
        root.append(filename)
        root.append(path)
        root.append(source)
        root.append(size)
        root.append(segmented)

        for ii in range(len(txtFile)):
            label = convertLabel(txtFile[ii][0])
            x_min_rect, x_max_rect, y_min_rect, y_max_rect = extractCoor(
                txtFile[ii], imgWidth, imgHeight)

            object = etree.Element("object")
            ####################object - element##################
            name = etree.SubElement(object, "name")
            name.text = "%s" % (label)

            pose = etree.SubElement(object, "pose")
            pose.text = "Unspecified"

            truncated = etree.SubElement(object, "truncated")
            truncated.text = "0"

            difficult = etree.SubElement(object, "difficult")
            difficult.text = "0"

            bndbox = etree.SubElement(object, "bndbox")
            #####sub_sub########
            xmin = etree.SubElement(bndbox, "xmin")
            xmin.text = "%d" % (x_min_rect)
            ymin = etree.SubElement(bndbox, "ymin")
            ymin.text = "%d" % (y_min_rect)
            xmax = etree.SubElement(bndbox, "xmax")
            xmax.text = "%d" % (x_max_rect)
            ymax = etree.SubElement(bndbox, "ymax")
            ymax.text = "%d" % (y_max_rect)
            #####sub_sub########

            root.append(object)
            ####################################################

        fileOutput = etree.tostring(root, pretty_print=True, encoding='UTF-8')
        # print(fileOutput.decode('utf-8'))
        ff = open(savePath + '%s.xml' % (imgName[:-4]), 'w', encoding="utf-8")
        ff.write(fileOutput.decode('utf-8'))


def yolo2xml(generalFolder, labelNames):
    global labels
    labels = labelNames
    if not os.path.exists(generalFolder + '/xml-labels'):
        os.mkdir(generalFolder + '/xml-labels')

    IMG_PATH = generalFolder + "/images/"
    fw = os.listdir(IMG_PATH)
    txtFolder = generalFolder + "/labels/"
    savePath = generalFolder + "/xml-labels/"
    yolo2xmlmain(IMG_PATH, fw, txtFolder, savePath)


def datumaro2poly():
    pass
