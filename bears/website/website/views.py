from django import forms
from django.views.generic.edit import FormView
from fastai.basic_train import load_learner
from fastai.vision.image import open_image
from django.conf import settings
from django.http import HttpResponse
from io import BytesIO
from django.shortcuts import render

learn = load_learner(settings.LEARN_FOLDER)

class ExaminatePicture(forms.Form):
    pic = forms.ImageField()

class ExaminateView(FormView):
    form_class = ExaminatePicture
    template_name = 'index.html'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        
        _,_,outputs = learn.predict(open_image(BytesIO(form.files['pic'].read())))
        predictions = zip(learn.data.classes, outputs)

        return render(self.request, self.template_name, {
            'form': form,
            'predictions': predictions
        })
        