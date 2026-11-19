from django.contrib.auth.decorators import user_passes_test

def is_admin(user):
    return user.groups.filter(name="Admin").exists()

def is_analyst(user):
    return user.groups.filter(name="Analyst").exists()

def is_viewer(user):
    return user.groups.filter(name="Viewer").exists()

def admin_required(view_func):
    return user_passes_test(is_admin)(view_func)

def analyst_or_admin_required(view_func):
    return user_passes_test(
        lambda u: is_admin(u) or is_analyst(u)
    )(view_func)
