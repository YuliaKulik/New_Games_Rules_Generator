import random
import textwrap
from flask import Flask, render_template, request
from model.model import generate
from model.text_processing import processing


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        prompt = request.form.get('input').capitalize()
        result = processing(textwrap.fill(generate(prompt), 150))
    return render_template('index.html', result=result, prompt=prompt)


if __name__ == '__main__':
    app.run()
