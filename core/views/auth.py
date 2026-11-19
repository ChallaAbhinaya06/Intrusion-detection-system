from django.contrib.auth.views import LoginView

class IDSLoginView(LoginView):
    template_name = "core/login.html"
