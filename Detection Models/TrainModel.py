# !pip install ultralytics
# !pip install roboflow

# from ultralytics import YOLO
from IPython.display import Image #display image file
from PIL import Image as im #read image file
# from roboflow import Roboflow

class TrainModel():
    '''
    Object Detection Class - Train the model with aerial images of flat roof model and perform predictions on new data    
    '''
    def __init__(self):
        pass

    def model_train(model, datafile):
        '''
        PARAMETERS -
        model : Any YOLOv8 model

        Training Model - With help of Roboflow, the respective dataset is imported and the model is trained. By default, the max epoch is set to 50, which can be changed
                         as per requirement. The model will return a validation test image to show the trained model with the confidence rate.
        '''
        # rf = Roboflow(api_key='l05N5YrhNgMFM59AKEVh', model_format='yolov8')
        # dataset = rf.workspace('ankit-badxt').project('flat-roof-with-objects').version(4).download(location='/content/my-dataset')
         # build from YAML and transfer weights
        # model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        results = model.train(data=datafile, epochs=50, batch=5)  # train the model
        return Image(filename='runs/detect/train/val_batch0_pred.jpg', width=800)   
        

    def predict_image(model, image_source, task):
        '''
        PARAMETERS -
        model : Any YOLOv8 model
        image_source : source of the image. The image could be a URL source, an existing image on the device.
        
        Image Prediction - With the trained model, predict the image. As we predict the image, we check if the model consist of annotated class - 𝘰𝘣𝘴𝘵𝘢𝘤𝘭𝘦.
        '''
        # model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        # test_image = image
        # test_image = im.open('/content/my-dataset/test/images/Roof-93_jpg.rf.c2380132b71c55e531c65b3b08d4fa27.jpg')
        predictions = model.predict(source=image_source, save=True)

        predicted_class = predictions[0].boxes.cls
        # print(predicted_class)
        if task == 'obstacle':
            if '[]' in str(predicted_class):
                return 'No obstacle detected. Select the roof'
            else:
                return 'Obstacle detected. Please find another roof'
        elif task == 'roof':
            if '1.' in str(predicted_class):
                print('There are no flat roofs available!')
            else:
                print('Flat Roof detected. Proceed to next task!')