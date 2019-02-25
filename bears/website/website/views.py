from django import forms
from django.views.generic.edit import FormView
from fastai.basic_train import load_learner
from django.conf import settings
from django.http import HttpResponse

learn = load_learner(settings.LEARN_FOLDER)

class ExaminatePicture(forms.Form):
    pic = forms.ImageField()

class ExaminateView(FormView):
    form_class = ExaminatePicture
    template_name = 'index.html'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        
        pred_class,_,losses = learner.predict(form.files['pic'].read())
        return HttpResponse(str(pred_class))
        