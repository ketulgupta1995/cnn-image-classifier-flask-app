from flask import Flask, render_template
import utils


app = Flask(__name__)


@app.route('/')
def hello():



    return render_template("index.html",
                           default_category="Bag", categories=['T-shirt/top',
                                                            'Trouser',
                                                            'Pullover',
                                                            'Dress',
                                                            'Coat',
                                                            'Sandal',
                                                            'Shirt',
                                                            'Sneaker',
                                                            'Bag',
                                                            'Ankle boot'])


if __name__ == '__main__':
    app.run()
