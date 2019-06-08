from flask import Flask, render_template, url_for, send_from_directory
import append_sys_path  # noqa

import minetorch
from minetorch import model, dataset, dataflow, loss, optimizer
from minetorch.web.api import api, experiment, graph


app = Flask(__name__, template_folder='dist', static_url_path='')
app.register_blueprint(api)
app.register_blueprint(experiment)
app.register_blueprint(graph)

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


@app.route('/static/<path:path>')
def send_assets(path):
    return send_from_directory('./dist/static', path)


if __name__ == '__main__':
    app.run()
