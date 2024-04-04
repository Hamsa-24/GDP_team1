from ultralytics import YOLO
from roboflow import Roboflow

class Roof_Detection():
    '''
    Roof Detection Class - Detect the flat roof from aerial images
    '''
    def roof_train():
        '''
        Training Model - With help of Roboflow, the respective dataset is imported and the model is trained. By default, the max epoch is set to 50, which can be changed
                         as per requirement. The model will return a validation test image to show the trained model with the confidence rate.
        '''
        rf = Roboflow(api_key='l05N5YrhNgMFM59AKEVh', model_format='yolov8')
        datasets = rf.workspace('ankit-badxt').project('roof-detection-kmib1').version(1).download(location='/content/my-dataset')
        datafile = '/content/my-dataset/data.yaml'
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        results = model.train(data=datafile, epochs=50, batch=5)  # train the model
    
    def roof_predict():
        '''
        Image Prediction - With the trained model, predict the image. As we predict the image, we check if the model consist of annotated class - 𝘰𝘣𝘴𝘵𝘢𝘤𝘭𝘦.
        '''
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        test_image = 'https://c7.alamy.com/comp/W3TMEC/aerial-photo-of-a-bungalow-settlement-in-essen-with-small-gardens-and-flat-roofs-in-essen-ruhr-area-north-rhine-westphalia-germany-essen-de-euro-W3TMEC.jpg'
        predictions = model.predict(source=test_image, save=True, conf=0.5)
        predicted_class = int(predictions[0].boxes.cls[0])
        if predicted_class == 0:
            print("Flat Roof detected. Please find below the bounding box (co-ordinates) of the region")
            a = int(predictions[0].boxes.xyxy[0][0]) #LEFT    # get box coordinates in (left, top, right, bottom) format
            b = int(predictions[0].boxes.xyxy[0][1]) #TOP
            c = int(predictions[0].boxes.xyxy[0][2]) #RIGHT
            d = int(predictions[0].boxes.xyxy[0][3]) #BOTTOM
            return a,b,c,d
        else:
            return "No Flat Roof detected"
        
class Obstacle_Detection():
    '''
    Obstacle Detection Class - Detect objects on the roof.
    '''
    def object_train():
        '''
        Training Model - With help of Roboflow, the respective dataset is imported and the model is trained. By default, the max epoch is set to 50, which can be changed
                         as per requirement. The model will return a validation test image to show the trained model with the confidence rate.
        '''
        rf = Roboflow(api_key='l05N5YrhNgMFM59AKEVh', model_format='yolov8')
        datasets = rf.workspace('ankit-badxt').project('flat-roof-with-objects').version(4).download(location='/content/my-dataset')
        datafile = '/content/my-dataset/data.yaml'
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        results = model.train(data=datafile, epochs=50, batch=5)  # train the model
    
    def object_predict():
        '''
        Image Prediction - With the trained model, predict the image. As we predict the image, we check if the model consist of annotated class - 𝘰𝘣𝘴𝘵𝘢𝘤𝘭𝘦.
        '''
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        test_image = 'https://roof-maker.co.uk/wp-content/uploads/2018/06/K92s4ExBFAwBW4c5KyNqnpdxO9HBOFyBG6nrDYp7tjBDFy76kdOytONfjMOt5HrXR-2.jpg'
        predictions = model.predict(source=test_image, save=True, conf=0.5)
        predicted_class = int(predictions[0].boxes.cls[0])
        if predicted_class == 0:
            return 'Obstacle detected. Please find another roof!'
        else:
            return 'No obstacle detected. Proceed selecting the roof!'

# if __name__ == '__main__':
#     count = 1

#     ask = input('Roof or Obstacle?')
#     if ask == 'roof':
#         if count == 1:
#             Roof_Detection.roof_train()
#             Roof_Detection.roof_predict()
#             count += 1
#         else:
#             Roof_Detection.roof_predict()
#             count += 1
#     elif ask == 'obstacle':
#         if count == 1:
#             Obstacle_Detection.object_train()
#             Obstacle_Detection.object_predict()
#             count += 1
#         else:
#             Obstacle_Detection.object_predict()
#             count += 1
