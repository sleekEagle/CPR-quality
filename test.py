from transformers import pipeline
from PIL import Image
import requests


pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
depth = pipe(image)["depth"]


from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import torch

import numpy as np

from PIL import Image

import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")

model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# prepare image for the model

inputs = image_processor(images=image, return_tensors="pt")