'''
    Roof dector/ classifier. This script imports an image corresponding to an aerial \
    image of a city and returns the list of the coordinates of the flat roofs as well \
    as their dimensions.

'''


from roof_detection.predict import get_prediction, segment_instance
import PIL
import numpy as np
from tensorflow.keras.models import load_model
import os

aerial = 'aerial_image.png'
image = PIL.Image.open(aerial)

roof_classifier = load_model('models/roof_classifier-1')
masks, boxes, pred_cls = get_prediction(aerial, confidence=0.5)


pred_cls = np.zeros(len(boxes))
for box in boxes:
    x_min, y_min, x_max, y_max = box
    roi = image.crop((x_min, y_min, x_max, y_max))
    roi = roi.resize((100,100))
    roi = np.expand_dims(np.array(roi), axis=0)

    rooftype = roof_classifier.predict(roi)[0]
    np.append(pred_cls, rooftype)
classes = np.where(pred_cls < 0.5, 0, 1)

image_path = os.path.join('~/Desktop/Cranfield/GDP/python/aerial_image.png')
segment_instance(image_path, masks=masks, boxes=boxes, pred_cls=pred_cls)
