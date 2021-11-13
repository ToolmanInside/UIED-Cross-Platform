from match_elements.GUI_pair import GUIPair
import cv2
from matplotlib import pyplot as plt
from random import randint as rint
from difflib import SequenceMatcher
import requests
import base64
import json
from logzero import logger


# gui = GUIPair('0001')
# gui.element_detection(True, False, False)

post_url = "http://192.168.50.94:6785/guimatching"
figure_1_path = "data/input/A0001.jpg"
figure_2_path = "data/input/I0001.jpg"

with open(figure_1_path, "rb") as f:
    base64_figure_1 = base64.b64encode(f.read()).decode('utf8')

with open(figure_2_path, "rb") as f:
    base64_figure_2 = base64.b64encode(f.read()).decode('utf8')

headers = {
    "Authorization": None,
    "Content-Type": "application/json"
}

form_data = json.dumps({
    "figure1": base64_figure_1,
    "figure2": base64_figure_2
})

logger.debug("send matching request")
response = requests.post(url=post_url, headers=headers, data=form_data).text
logger.debug("respose received")