from transformers import pipeline
from PIL import Image
import requests

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
depth = pipe(image)["depth"]

depth.show()



