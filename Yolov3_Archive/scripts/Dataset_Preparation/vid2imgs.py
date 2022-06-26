import cv2
import os
import argparse
import sys
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Convert video to images. [NOTE] all files in the output folder will be deleted, make sure you have backup.')
parser.add_argument('-i', type=str, help='input video path')
parser.add_argument('-o', type=str, help='output FOLDER path')
parser.add_argument('-f', type=str, help='output images format (such as png/jpg)')
parser.add_argument('-p', type=int,
                    help='[OPTIONAL] how many images per folder. By default all (all video frames will be saved).',
                    default='-1')
parser.add_argument('-s', type=tuple, default=(0, 0), help='[OPTIONAL] resize output frames. Tuple format please.')

args = parser.parse_args()


def main(inpVideoPath, outFolderPath, outFormat, seconds, resize):
    print(inpVideoPath)
    cap = cv2.VideoCapture(inpVideoPath)
    if not cap.isOpened():
        print("could not open :", inpVideoPath)
        sys.exit()

    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if seconds == -1:
        seconds = numFrames

    NUM_OUTPUT_VIDEOS = int(numFrames / int(seconds))
    FRAMES_PER_VIDEO = int(seconds)
    OUTPUT_VIDEO_SIZE = resize if resize[0] != 0 else (width, height)
    print('[video loaded succesfully]')

    if os.path.exists(outFolderPath):
        shutil.rmtree(outFolderPath)

    outs = {}

    os.mkdir(outFolderPath)

    for i in range(NUM_OUTPUT_VIDEOS):
        os.mkdir(os.path.join(outFolderPath, str(i)))
        # outs[i] = cv2.VideoWriter( os.path.join( outFolderPath,str(i)+'.'+ outFormat), cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize=OUTPUT_VIDEO_SIZE )
    print('[output folder & subfolders created]')

    j = 0

    print('[saving images]')
    for i in tqdm(range(numFrames)):

        if i % FRAMES_PER_VIDEO == 0 and i != 0:
            # outs[j].release()
            j += 1
            if j == NUM_OUTPUT_VIDEOS:
                j -= 1

        ret, frame = cap.read()
        frame = cv2.resize(frame, OUTPUT_VIDEO_SIZE)

        imgName = format(i, '05') + '.' + outFormat
        fullSavePath = os.path.join(outFolderPath, imgName)

        cv2.imwrite(fullSavePath, frame)
    print('[done!]')


if __name__ == '__main__':
    main(args.i, args.o, args.f, args.p, args.s)
