import json
from flask import Flask, jsonify, render_template
from config import config


def create_app():
    app = Flask(__name__)
    app.config.update(config.as_flask_config_dict())
    return app

app = create_app()


@app.route('/')
def home():
    return jsonify({"message": "Hello Retinex Team"})
    

@app.route('/evaluation/')
def evaluate():
    return render_template('evaluate.html', result="aa")

if __name__ == '__main__':
    app.run()