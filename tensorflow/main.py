from flask import Flask, request, Response
import numpy as np
import cv2
import json
import argparse


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

#from utilities import load_model, load_label_map, show_inference, parse_output_dict
from custom_np_encoder import NumpyArrayEncoder

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/drive"]

DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'

#model_path = "models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model"
#labels_path = "data/mscoco_label_map.pbtxt"

#vis_threshold = 0.5
#max_boxes = 20

#detection_model = load_model(model_path):
#category_index = load_label_map(labels_path)

def parse_args():

  parser = argparse.ArgumentParser(description='Tensorflow object detection API')

  parser.add_argument('--debug', dest='debug',
                        help='Run in debug mode.',
                        required=False, action='store_true', default=False)

  parser.add_argument('--port', dest='port',
                        help='Port to run on.', type=int,
                        required=False, default=DEFAULT_PORT)

  parser.add_argument('--host', dest='host',
                        help='Host to run on, set to 0.0.0.0 for remote access', type=str,
                        required=False, default=DEFAULT_HOST)

  args = parser.parse_args()
  return args

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/object_detection', methods=['POST'])
def infer():
    print("infer")
    # convert image data to uint8
    nparr = np.frombuffer(request.data, np.uint8)
    return processImage(nparr)


# route http posts to this method
@app.route('/getobject_detection', methods=['POST'])
def getinfer():

  print("getinfer")
  data = request.get_json()
  username = data["username"]
  image = data["photo"]
  print(username)
  print(image)

  service_account_json_key = './token.json'
  credentials = service_account.Credentials.from_service_account_file(
                              filename=service_account_json_key,
                              scopes=SCOPES)
  print("before try")

  # need to do
  # extract request.data as drive URL and extract file_id

  try:
    service = build('drive', 'v3', credentials=credentials)

    print(service)

    file_id = "1Zu0H50xy1UEQ0sw12MZiZBrV3mo_GGpY"
    #file_id = "1_KvDCHo_auEH8e1nuipSm3W2_ppenjB2"
    res = service.files().get_media(fileId=file_id).execute()
    
    print(res[0])

    nparr = np.frombuffer(res, np.uint8)
   
    print("to process image")

    return processImage(nparr)

  except HttpError as error:
    # TODO(developer) - Handle errors from drive API.
    print(f"An error occurred: {error}")
    return Response(response="0", status=500, mimetype="application/json")


def getModel():

  save_model = 'vgg16_weight.h5'
  num_classes = 2
  # Load VGG16 model without the top layers
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

  # Add custom layers on top
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(num_classes, activation='softmax')(x)  
  # num_classes = number of eye disease classes

  # Create the final model
  load_final_model = Model(inputs=base_model.input, outputs=predictions)
  # Load the weights
  load_final_model.load_weights(save_model)
  
  return load_final_model


def predictImage(img):
  img_array = preprocess_input(img)
  final_model = getModel()
  out = final_model.predict(img_array)
  return out[0]


def processImage(nparr):
    figsize = (224,224)
    # decode image
    imgo = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(imgo, figsize)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    result = predictImage(array)
    
    parsed_dict = {}
 
    # add the size of the image in the response
    parsed_dict.update({"image size": "size={}x{}".format(imgo.shape[1], imgo.shape[0])})
   
    parsed_dict.update({"result": "prob={} : {}".format(result[0], result[1])})


    # build a response dict to send back to client
    response = parsed_dict
    
    # encode response
    response_encoded = json.dumps(response, cls=NumpyArrayEncoder)

    return Response(response=response_encoded, status=200, mimetype="application/json")


# start flask app
def main():
  args = parse_args()
  app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
