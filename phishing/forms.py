from django import forms
from .models import CallHistory

class AudioDataForm(forms.ModelForm):
    class Meta:
        model = CallHistory
        fields = ['audio_file']