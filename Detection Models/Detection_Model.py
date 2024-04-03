from ultralytics import YOLO
from TrainModel import TrainModel
from roboflow import Roboflow

# count = 1
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

def obstacle_detection():
    rf = Roboflow(api_key='l05N5YrhNgMFM59AKEVh', model_format='yolov8')
    dataset = rf.workspace('ankit-badxt').project('flat-roof-with-objects').version(4).download(location='/content/my-dataset')
    datafile = '/content/my-dataset/data.yaml'
    return datafile

def roof_detection():
    rf = Roboflow(api_key='l05N5YrhNgMFM59AKEVh', model_format='yolov8')
    dataset = rf.workspace('ankit-badxt').project('roof-detection-kmib1').version(1).download(location='/content/my-dataset')
    datafile = '/content/my-dataset/data.yaml'
    return datafile


task = input('Do you want to detect roof or obstacle?')

if task == 'roof':
    TrainModel.model_train(model, roof_detection())
    test_image = 'https://c7.alamy.com/comp/W3TMEC/aerial-photo-of-a-bungalow-settlement-in-essen-with-small-gardens-and-flat-roofs-in-essen-ruhr-area-north-rhine-westphalia-germany-essen-de-euro-W3TMEC.jpg'
    TrainModel.predict_image(model, test_image)
elif task == 'obstacle':
    TrainModel.model_train(model, obstacle_detection(), task)
    test_image = 'https://roof-maker.co.uk/wp-content/uploads/2018/06/K92s4ExBFAwBW4c5KyNqnpdxO9HBOFyBG6nrDYp7tjBDFy76kdOytONfjMOt5HrXR-2.jpg'
    TrainModel.predict_image(model, test_image)

# if count == 1:
#     TrainObjectDetection.model_train(model)
#     TrainObjectDetection.predict_image(model, test_image)
#     count += 1
# else:
#     TrainObjectDetection.predict_image(model, test_image)
