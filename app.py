import base64
import random
from io import BytesIO

import flask
import torch
import utils
from PIL import Image, ImageOps
from flask import Flask, render_template
from torchvision import transforms

app = Flask(__name__)


@app.route('/userimage', methods=["POST"])
def testuserimage():
    imagefile = flask.request.files.get('imagefile', '')
    image_string = base64.b64encode(imagefile.read())

    img = Image.open(BytesIO(base64.b64decode(image_string)))
    # img.show()
    simg = img.resize((28, 28)).convert('L')
    simg_inverted = ImageOps.invert(simg)
    pil_to_tensor_inverted = transforms.ToTensor()(simg_inverted)
    im_inverted = pil_to_tensor_inverted.view(28, 28).numpy() * 255
    processed_im = torch.from_numpy(im_inverted.astype("uint8"))
    processed_im = processed_im.view(1, 28, 28)

    with torch.no_grad():
        outputs = utils.myNN(processed_im.float())
        _, predicted = torch.max(outputs.data, 1)
    return render_template("index.html",
                           default_category="Bag", categories=utils.categories, images=utils.images, form_image_prediction =utils.test_set.classes[predicted])
    # return "" + utils.test_set.classes[predicted] + ""


@app.route('/')
def hello():
    utils.images.clear()
    for i in range(0, 10):
        rand_num = random.randint(1, 8000)
        utils.images[rand_num] = []
        image = Image.fromarray(utils.test_set.data[rand_num].numpy())
        buffered = BytesIO()
        image.resize((200, 200)).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        utils.images[rand_num].append(img_str.decode("utf-8"))
        utils.images[rand_num].append(utils.test_set.classes[utils.test_set.targets[rand_num]])
        with torch.no_grad():
            outputs = utils.myNN(utils.test_set.data[rand_num].view(1, 28, 28).float())
            _, predicted = torch.max(outputs.data, 1)
        utils.images[rand_num].append(utils.test_set.classes[predicted])
    return render_template("index.html",
                           default_category="Bag", categories=utils.categories, images=utils.images,
                           form_image_prediction="")


if __name__ == '__main__':
    app.run()
