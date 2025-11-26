from django.urls import path
from .api import api
from .views import fetch_library_version

urlpatterns = [
    path("", api.urls),
    path('fetch-library-version/', fetch_library_version, name='fetch-library-version'),
]