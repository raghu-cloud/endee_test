from ninja import NinjaAPI
# from .utils import run_test
# from .models import TestRun
# from django.shortcuts import get_object_or_404
# import threading
from django.http import JsonResponse
import subprocess
import sys
from .schemas import RunTestRequest, UpdateLibraryVersionRequest

api = NinjaAPI()

# @api.post("/run-tests/")
# def run_tests(request, test_run_id: int):
#     """
#     API to trigger a test run.
#     Pass `test_run_id` of the TestRun object in DB.
#     """
#     try:
#         # Fetch the TestRun DB object
#         test_run = get_object_or_404(TestRun, id=test_run_id)

#         # Optional: allow passing API key from request if needed
#         api_key = request.headers.get("X-API-KEY")

#         # Run in background
#         threading.Thread(target=run_test, args=(test_run, api_key)).start()

#         return {"status": "success", "message": f"Test {test_run.id} triggered"}
    
#     except Exception as e:
#         return {"status": "error", "error": str(e)}

@api.post("/run-tests/")
def run_tests(request, payload: RunTestRequest):
    test_run_id = payload.test_run_id
    api_key = request.headers.get("X-API-KEY", "")

    try:
        # Run subprocess to execute test
        command = [sys.executable, 'manage.py', 'run_test_job', str(test_run_id)]
        if api_key:
            command += ['--api_key', api_key]

        subprocess.Popen(command)
        return {"status": "success", "message": f"Test run #{test_run_id} started on Windows server."}

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
@api.post("/install-library/")
def install_library(request, payload: UpdateLibraryVersionRequest):
    version_update_id = payload.version_update_id

    try:
        # Run subprocess to execute test
        command = [sys.executable, 'manage.py', 'install_library', str(version_update_id)]

        subprocess.Popen(command)
        return {"status": "success", "message": f"Library updating for  #{version_update_id} started on Windows server."}
    
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


