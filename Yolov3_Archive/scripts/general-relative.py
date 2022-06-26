from utils.utils import *
from utils.train import train
from utils.test import test
from utils.detect import detect
from utils.prune import prune
from utils.plot_prune_graphs import plotPruneGraphs
from utils.img_size_full_test import imgSizeTest
from utils.img_size_plot_multi_graph import imgSizePlotGraph

import gen_anchor

import os

# MAKROS
CONFIG_PATH = './config.cfg'

if __name__ == '__main__':
    configDict = parseConfig(CONFIG_PATH)

    if configDict['general']['train']:
        configDict['train']['cfg'] = findModelCfgPath(configDict['train']['model-path'])

        if configDict['train']['calculate_anchors']:
            numClustures = configDict['train']['anchor-custer-num']
            width = configDict['train']['anchor-width']
            height = configDict['train']['anchor-height']
            gen_anchor.main(os.path.join(configDict['train']['model-path'], 'train.txt'), numClusterss=numClustures,
                            width=width, height=height)
            changeAnchorsInModelCfg(numClustures, configDict['train']['cfg'])

        if configDict['train']['create-data-file']:
            prepareData(configDict['train']['model-path'])
            configDict['train']['data'] = os.path.join(configDict['train']['model-path'], 'data.data')

        train(configDict['train'])

        pt2weights(configDict['train']['cfg'])
        moveWeightsToModelPath(configDict['train']['model-path'])
        moveResultsToModelPath(configDict['train']['model-path'])

    if configDict['general']['test']:
        configDict['test']['cfg'] = findModelCfgPath(configDict['test']['model-path'])
        configDict['test']['weights'] = os.path.join(configDict['test']['model-path'], "best.pt")

        test(configDict['test'])

    if configDict['general']['detect']:
        modelName = configDict['detect']['model-path'].split('/')[-1]
        imgSize = configDict['detect']['img-size']
        outputPath1 = f"../model_detect_outputs/{modelName}/"
        outputPath2 = f"../model_detect_outputs/{modelName}/{modelName}_{imgSize}/"

        configDict['detect']['cfg'] = findModelCfgPath(configDict['detect']['model-path'])
        configDict['detect']['names'] = os.path.join(configDict['detect']['model-path'], "obj.names")
        configDict['detect']['weights'] = os.path.join(configDict['detect']['model-path'], "best.pt")
        configDict['detect']['output'] = outputPath2

        mkdirDetectOutputFolder(outputPath1, True)
        mkdirDetectOutputFolder(outputPath2)

        detect(configDict['detect'])

    if configDict['general']['prune']:
        if not configDict['prune']['prune-only-draw-graph']:
            prune(configDict['test'], configDict['prune'])
        plotPruneGraphs(configDict['prune'])

    if configDict['general']['img-size-test']:
        if not configDict['img-size-test']['img-size-test-only-draw-graph']:
            imgSizeTest(configDict['img-size-test'], configDict['test'])
        imgSizePlotGraph(configDict['img-size-test'])
