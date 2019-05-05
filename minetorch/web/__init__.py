from flask import Flask, render_template, url_for
from flask.json import JSONEncoder

import minetorch
from minetorch import model, dataset, dataflow, loss, optimizer
from .api import api, experiment


app = Flask(__name__)

app.register_blueprint(api)
app.register_blueprint(experiment)

minetorch.core.boot()

@app.route('/')
def index():
    return render_template(
        'index.html',
        url_for=url_for,
        model=model,
        dataset=dataset,
        dataflow=dataflow,
        loss=loss,
        optimizer=optimizer)
