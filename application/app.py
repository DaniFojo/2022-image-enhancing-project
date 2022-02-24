from flask import Flask, render_template, redirect, url_for
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


@app.route('/upload_image/', methods=['GET','POST'])
def upload_image():
    form = ImageForm()
    if form.validate_on_submit():
        image = form.image.data
        image.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static/uploads', image.filename))
        return render_template('upload_image.html', form=form, filename=image.filename)
    return render_template('upload_image.html', form=form)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run()