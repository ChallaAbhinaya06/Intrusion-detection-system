from django import forms


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class DatasetUploadForm(forms.Form):
    datasets = forms.FileField(
        required=False
    )
