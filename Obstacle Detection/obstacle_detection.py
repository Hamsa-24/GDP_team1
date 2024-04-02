from ultralytics import YOLO
from TrainObjectDetection import TrainObjectDetection

# count = 1
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights


test_image = 'https://roof-maker.co.uk/wp-content/uploads/2018/06/K92s4ExBFAwBW4c5KyNqnpdxO9HBOFyBG6nrDYp7tjBDFy76kdOytONfjMOt5HrXR-2.jpg'

TrainObjectDetection.model_train(model)
TrainObjectDetection.predict_image(model, test_image)
# if count == 1:
#     TrainObjectDetection.model_train(model)
#     TrainObjectDetection.predict_image(model, test_image)
#     count += 1
# else:
#     TrainObjectDetection.predict_image(model, test_image)