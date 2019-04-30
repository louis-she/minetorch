from flask import Flask, render_template, url_for
from minetorch import model
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', url_for=url_for, model=model)
