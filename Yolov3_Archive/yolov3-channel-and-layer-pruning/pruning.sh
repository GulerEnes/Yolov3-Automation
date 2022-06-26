#bash
python3 /jup/tmp/python_scripts/yolov3-channel-and-layer-pruning/prune.py \
--cfg /jup/tmp/python_scripts/yolov3_archive/yolov3/comparable_tests/$1/$1.cfg \
--data /jup/tmp/python_scripts/yolov3_archive/yolov3/comparable_tests/$1/$1.data \
--weights /jup/tmp/python_scripts/yolov3_archive/yolov3/weights/best_$1.pt \
--percent $2 #\
#--img-size 416
