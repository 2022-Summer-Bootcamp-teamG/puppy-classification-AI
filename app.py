from flask import render_template, request, Flask
from predict import *

app = Flask(__name__)


@app.route('/')
def hello():
    return "Run Flask"


if __name__ == '__main__':
    app.run()
