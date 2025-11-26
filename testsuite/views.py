from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .utils import get_library_version

# Create your views here.

@require_GET
def fetch_library_version(request):
    lib = request.GET.get('library')
    version = get_library_version(lib)
    return JsonResponse({'version': version})