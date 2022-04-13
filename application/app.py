import os
import sys
from flask import Flask, render_template, redirect, url_for, request
from config import config
from forms import ImageForm
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # Necesario para importar módulo desde el parent folder.
from utils import create_directory_if_not_exists, load_model


def create_app():
    application = Flask(__name__)
    application.config.update(config.as_flask_config_dict())
    return application


app = create_app()

model_decom_split = None
model_relight_split = None


def load_model_pair(decom_pt_file_name, relight_pt_file_name):
    base_path = os.path.join('/opt', 'proj_img_enhance', 'checkpoints')
    path_decomposition = os.path.join(base_path, decom_pt_file_name)
    path_relight = os.path.join(base_path, relight_pt_file_name)

    print("path_decomposition: ", path_decomposition)
    print("path_relight: ", path_relight)

    print("Loading models...")
    mdl_decom = load_model(path_decomposition, 'decom', 'cpu')
    mdl_rel = load_model(path_relight, 'relight', 'cpu')

    # Set eval mode:
    mdl_decom.eval()
    mdl_rel.eval()

    torch.set_grad_enabled(False)

    print("Models loaded.")

    return mdl_decom, mdl_rel



def enhance_image(path_img_low, model_decom, model_relight):

    sample_low = Image.open(path_img_low)
    orig_img_size = (sample_low.size)
    print("orig size: ", orig_img_size)

    # Aplicar image enhancing:
    transform = transforms.Compose(
        [transforms.Resize([297, 297]), transforms.ToTensor()])    
    sample_low = transform(sample_low)    
    # Add batch dimension:
    sample_low = sample_low.unsqueeze(0)
    print("sample_low shape: ", sample_low.shape)

    img_reflectance, img_illuminance = model_decom(sample_low)
    i_enhanced = model_relight(torch.concat((img_reflectance, img_illuminance), dim=1))

    i_enhanced_3 = torch.concat((i_enhanced, i_enhanced, i_enhanced), dim=1)
    img_reconstructed = img_reflectance * i_enhanced_3

    print("img_reconstructed: ", img_reconstructed.shape)
    
    # Discard batch dimension:
    img_reconstructed = img_reconstructed.squeeze()

    # Convertir tensor a png:
    pil_img = to_pil_image(img_reconstructed)
    pil_img = pil_img.resize(orig_img_size)

    return pil_img



# The code in this function will be executed before we recieve any request
@app.before_first_request
def _load_models():

    global model_decom_split, model_relight_split

    model_decom_split, model_relight_split = load_model_pair(os.path.join('split', 'model_decom_split.pt'), 
                                                             os.path.join('split', 'model_relight_split.pt'))



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

        reconstructed_img = enhance_image(path_img_low, model_decom_split, model_relight_split)
        reconstructed_img.save(os.path.join(path_uploads, "reconst_split.png"))

        # Mostrar resultados:
        return render_template('upload_image.html', form=form, filename=image.filename, img_enhanced="reconst_split.png")

    # Mostrar pantalla para subir imagen:
    return render_template('upload_image.html', form=form)



@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



if __name__ == '__main__':
    app.run()
