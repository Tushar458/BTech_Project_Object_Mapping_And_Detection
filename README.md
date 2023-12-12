# BTech_Project_Object_Mapping_And_Detection


In recent years, the global landscape has witnessed a profound transformation in Artificial Intelligence, driven largely by the remarkable strides made in deep learning. This paradigm shift has been particularly influential in the field of object detection, where cutting-edge algorithms have played a pivotal role in shaping the capabilities of AI systems. Among the array of notable object detection algorithms, Region-based Convolutional Neural Networks (RCNN), Faster-RCNN, Single Shot Detector (SSD), and You Only Look Once (YOLO) have garnered significant attention.

Faster-RCNN and SSD stand out as frontrunners in the pursuit of heightened accuracy in object detection tasks. Their design contributes towards superior precision, making them ideal choices in scenarios where fine-grained detection is paramount. Conversely, YOLO emerges as a powerhouse in situations where speed takes precedence over absolute precision. The You Only Look Once algorithm has therefore been used to successfully implement multiple object detection. Dataset used for this phase is the KITTI dataset, featuring eighty classes. However, this project focused on five specific classes: car, bus, truck, motorcycle, and train. Taking a step further, the application of multiple object detection principles seamlessly transitions into object tracking. This system was therefore was made using OpenCV and Python using YOLOv3 algorithm.

In essence, the intricate interplay between these advanced multiple-object detection algorithms as well as the delicate balance between accuracy and speed has found significant applications especially in the field of vehicle-mapping and subsequent traffic surveillance.

Here we setup the CNN and YOLO models to analyse them on the basis of various techniques such as Confusion matrix, Accuracy and loss, F1 score, recall and precision



Setup
Clone repo, install dependencies and check PyTorch and GPU.


!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt  # install dependencies

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
     
Setup complete. Using torch 1.9.0+cu102 (Tesla V100-SXM2-16GB)

detect.py runs YOLOv5 inference on a variety of sources, downloading models automatically from the latest YOLOv5 release, and saving results to runs/detect. Example inference sources are:

# python detect.py --source 0  # webcam
                          file.jpg  # image 
                          file.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/NUsoVlDFqZg'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
Image(filename='runs/detect/exp/zidane.jpg', width=600)
     
detect: weights=['yolov5s.pt'], source=data/images/, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False
YOLOv5 🚀 v5.0-367-g01cdb76 torch 1.9.0+cu102 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Fusing layers... 
Model Summary: 224 layers, 7266973 parameters, 0 gradients
image 1/2 /content/yolov5/data/images/bus.jpg: 640x480 4 persons, 1 bus, 1 fire hydrant, Done. (0.007s)
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.007s)
Results saved to runs/detect/exp
Done. (0.091s)

# Validate
Validate a model's accuracy on COCO val or test-dev datasets. Models are downloaded automatically from the latest YOLOv5 release. To show results by class use the --verbose flag. Note that pycocotools metrics may be ~1% better than the equivalent repo metrics, as is visible below, due to slight differences in mAP computation.

COCO val2017
Download COCO val 2017 dataset (1GB - 5000 images), and test model accuracy.


# Download COCO val2017
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../datasets && rm tmp.zip
     
  0%|          | 0.00/780M [00:00<?, ?B/s]

# Run YOLOv5x on COCO val2017
!python val.py --weights yolov5x.pt --data coco.yaml --img 640 --iou 0.65 --half
     
val: data=./data/coco.yaml, weights=['yolov5x.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65, task=val, device=, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True
YOLOv5 🚀 v5.0-367-g01cdb76 torch 1.9.0+cu102 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt to yolov5x.pt...
100% 168M/168M [00:08<00:00, 20.6MB/s]

COCO test-dev2017
Download COCO test2017 dataset (7GB - 40,000 images), to test model accuracy on test-dev set (20,000 images, no labels). Results are saved to a *.json file which should be zipped and submitted to the evaluation server at https://competitions.codalab.org/competitions/20794.


# Download COCO test-dev2017
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../ && rm tmp.zip # unzip labels
!f="test2017.zip" && curl http://images.cocodataset.org/zips/
f && unzip -q f && rm
f  # 7GB,  41k images
%mv ./test2017 ../coco/images  # move to /coco
     

# Run YOLOv5s on COCO test-dev2017 using --task test
!python val.py --weights yolov5s.pt --data coco.yaml --task test

3. Train
Download COCO128, a small 128-image tutorial dataset, start tensorboard and train YOLOv5s from a pretrained checkpoint for 3 epochs (note actual training is typically much longer, around 300-1000 epochs, depending on your dataset).


# Download COCO128
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../datasets && rm tmp.zip
     
Train a YOLOv5s model on COCO128 with --data coco128.yaml, starting from pretrained --weights yolov5s.pt, or from randomly initialized --weights '' --cfg yolov5s.yaml. Models are downloaded automatically from the latest YOLOv5 release, and COCO, COCO128, and VOC datasets are downloaded automatically on first use.

All training results are saved to runs/train/ with incrementing run directories, i.e. runs/train/exp2, runs/train/exp3 etc.

train the model using train.py on dataset and detect using detect.py
