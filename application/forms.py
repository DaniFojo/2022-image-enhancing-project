from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms.fields import SubmitField, FileField


class ImageForm(FlaskForm):
    image = FileField('image', validators=[FileRequired(),
                      FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('Start!')
