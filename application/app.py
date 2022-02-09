from flask import Flask, render_template
from config import config
from forms import ImageForm
import os


def create_app():
    app = Flask(__name__)
    app.config.update(config.as_flask_config_dict())
    return app

app = create_app()


@app.route('/')
def home():
    return render_template('base.html', message="Hello Retinex Team")


@app.route('/evaluation/', methods=['GET','POST'])
def evaluate():
    form = ImageForm()
    if form.validate_on_submit():
        f = form.image.data
        # f.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images', f.filename))
        return render_template('evaluate.html', form=form)

    return render_template('evaluate.html', form=form)

if __name__ == '__main__':
    app.run()