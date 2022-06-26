#bash
python3 layer_prune.py \
--cfg /jup/tmp/python_scripts/yolov3_archive/yolov3/comparable_tests/$1/$1.cfg \
--data /jup/tmp/python_scripts/yolov3_archive/yolov3/comparable_tests/$1/$1.data \
--weights /jup/tmp/python_scripts/yolov3_archive/yolov3/weights/best_$1.pt \
--shortcuts $2 #\
#--img-size 416
