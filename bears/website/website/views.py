from io import BytesIO

from django import forms
from django.views.generic.edit import FormView
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from fastai.basic_train import load_learner
from fastai.vision.image import open_image


LEARN = load_learner(settings.LEARN_FOLDER)


class ExaminatePicture(forms.Form):
    pic = forms.ImageField()

class ExaminateView(FormView):
    form_class = ExaminatePicture
    template_name = 'index.html'

    def form_valid(self, form):
        
        _,_,outputs = LEARN.predict(open_image(BytesIO(form.files['pic'].read())))
        predictions = zip(LEARN.data.classes, outputs)

        return render(self.request, self.template_name, {
            'form': form,
            'predictions': predictions
        })
        