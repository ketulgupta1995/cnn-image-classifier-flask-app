import random

import torch
from PIL import Image
from flask import Flask, render_template
import utils
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)


@app.route('/')
def hello():

    for i in range(0, 10):
        rand_num  = random.randint(1, 8000)
        utils.images[rand_num] = []
        image = Image.fromarray(utils.test_set.data[rand_num].numpy())
        buffered = BytesIO()
        image.resize((200,200)).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        utils.images[rand_num].append(img_str.decode("utf-8"))
        utils.images[rand_num].append(utils.test_set.classes[utils.test_set.targets[rand_num]])
        with torch.no_grad():
            outputs = utils.myNN(utils.test_set.data[rand_num].view(1, 28, 28).float())
            _, predicted = torch.max(outputs.data, 1)
        utils.images[rand_num].append(utils.test_set.classes[predicted])
    return render_template("index.html",
                               default_category="Bag", categories=utils.categories, images= utils.images)


if __name__ == '__main__':
    app.run()
