#!/usr/bin/python
                                                  
import numpy as np
import cv2
import requests
import json

    
def main():
  image_file = "/home/wojciech/Downloads/pred.png"
  # Convert arbitrary sized jpeg image to 28x28 b/w image.
  
  data = cv2.imread(image_file)
  data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
  data = cv2.resize(data, (400, 400))
  data = np.asarray(data, dtype=np.float32).reshape(-1, 400, 400, 3) / 255.0


  json_request = json.dumps({"instances": data.tolist()})

  resp = requests.post('http://localhost:8501/v1/models/UNet:predict', data=json_request)
  print('response.status_code: {}'.format(resp.status_code))     
  
  out = np.array(
                 
main()
