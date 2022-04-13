import os
import sys
from flask import Flask, render_template, redirect, url_for, request
from config import config
from forms import ImageForm
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import create_directory_if_not_exists, load_model


def create_app():
    application = Flask(__name__)
    application.config.update(config.as_flask_config_dict())
    return application


app = create_app()

model_decomposition = None
model_relight = None


# The code in this function will be executed before we recieve any request
@app.before_first_request
def _load_model():

    path_self = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    path_parent = os.path.dirname(path_self)
    print("path_self: ", path_self)
    print("path_parent: ", path_parent)

    # path_decomposition = os.path.join('checkpoints', 'split', 'decomposition', '2022_04_07_13_15_25','model_decomposition_epoch_1.pt')
    path_decomposition = os.path.join(path_parent, 'checkpoints', 'decomposition', 
                                        '2022_04_09_15_27_49','model_decomposition_epoch_100.pt')

    # path_relight = os.path.join('checkpoints', 'split', 'relight', '2022_04_07_13_15_25', 'model_relight_epoch_1.pt')
    path_relight = os.path.join(path_parent, 'checkpoints', 'relight', '2022_04_09_15_27_49', 'model_relight_epoch_100.pt')

    print("path_decomposition: ", path_decomposition)
    print("path_relight: ", path_relight)

    global model_decomposition, model_relight

    model_decomposition = load_model(path_decomposition, 'decom', 'cpu')
    model_relight = load_model(path_relight, 'relight', 'cpu')

    print("modelos cargados.")

    # eval_decom(model_decomposition, test_data_loader)
    # eval_relight(model_decomposition, model_relight, test_data_loader)




@app.route('/')
def home():
    return render_template('base.html', message="Hello Retinex Team")


@app.route('/upload_image/', methods=['GET', 'POST'])
def upload_image():

    """Permite subir una imagen y luego la muestra."""

    form = ImageForm()
    if form.validate_on_submit():
        image = form.image.data
        
        # Guardar imagen:
        path_self = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        create_directory_if_not_exists(os.path.join(path_self, 'static', 'uploads'))
        path_uploads = os.path.join(path_self, 'static', 'uploads')
        path_img_low = os.path.join(path_uploads, image.filename)
        image.save(path_img_low)

        # Aplicar image enhancing:
        sample_low = Image.open(path_img_low)
        transform = transforms.Compose(
            [transforms.Resize([300, 300]), transforms.ToTensor()])    
        sample_low = transform(sample_low)    
        # Add batch dimension:
        sample_low = sample_low.unsqueeze(0)
        print("sample_low shape: ", sample_low.shape)

        img_reflectance, img_illuminance = model_decomposition(sample_low)
        print("img_reflectance: ", img_reflectance.shape)
        print("img_illuminance: ", img_illuminance.shape)
        # Quitar dimensión del batch:
        img_illuminance = img_illuminance.squeeze()

        illuminance_pil = to_pil_image(img_illuminance)
        illuminance_pil.save(os.path.join(path_uploads, "illum.png"))

        # Mostrar resultados:
        return render_template('upload_image.html', form=form, filename=image.filename, img_enhanced="illum.png")

    # Mostrar pantalla para subir imagen:
    return render_template('upload_image.html', form=form)



@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



if __name__ == '__main__':
    app.run()
