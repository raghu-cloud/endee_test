import subprocess
import json
import os
import sys
import time
from datetime import datetime
from .models import TestRun, get_platform, LibraryUpdateRequest
from vecx.vectorx import VectorX
import re
from pathlib import Path
import pkg_resources
from importlib.metadata import version, PackageNotFoundError


def run_test(db_object: TestRun, api_key = None):
    """
    Execute the appropriate test based on the test_type selected in TestRun
    """
    if get_platform() == "windows":
        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "_")
    else:
        timestamp = datetime.now().isoformat()


    test_type_code = db_object.test_type
    report_file = f"report_testcode{test_type_code}_{timestamp}.json"

    # if api_key != None:
    #     with open("api_token_file.txt", 'w') as f:
    #         f.write(api_key)

    # Map test types to their corresponding test file paths
    test_mapping = {
        # 1: "testsuite/tests/test_01_create_index.py",
        # 2: "testsuite/tests/test_02_upsert.py",
        # 3: "testsuite/tests/test_03_query_index.py",
        # 4: "testsuite/tests/test_04_delete_vectors_index.py",
        1: "testsuite/tests/",
        2: "testsuite/tests_hybrid_index/",
        3: "testsuite/tests_vectorx_langchain_integration/",
        4: "testsuite/tests_vectorx_llamaindex_integration/",
        5: "testsuite/tests_vectorx_crewai_integration/"
    }

    with open(f"config/pipeline_mode_bool_{timestamp}.txt", "w") as f:
        f.write(str(db_object.pipeline_mode))
    test_path = test_mapping.get(test_type_code)
    if not test_path:
        raise ValueError(f"Unknown test type code: {test_type_code}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    if get_platform() == "windows":
        command = [
            sys.executable,
            "-m", "pytest",
            "-s",
            "-o", "log_cli=true",
            "--json-report",
            f"--json-report-file={report_file}",
            test_path
        ]
    else:
        # Run pytest command
        command = [
            "PYTHONPATH=.",
            "pytest",
            "-s",
            "-o", "log_cli=true",
            "--json-report",
            f"--json-report-file={report_file}",
            test_path
        ]

    if api_key:
        command.extend(["--api-key", api_key])

    
    command.extend(["--timestamp", timestamp])

    print("Thread running")
    db_object.status = 'running'
    db_object.save()
    # try:
    #     # Join the command list into a string for shell execution
    #     cmd_str = " ".join(command)
    #     process = subprocess.run(cmd_str, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, capture_output=True, text=True)
    #     for line in process.stdout:
    #         print(line, end="")  # stream output live to Django server console
    #     process.wait()
    #     print("subprocess ends")
    #     try:
    #         with open(report_file) as f:
    #             report_data = json.load(f)  # Load JSON into a dict
    #             db_object.report = report_data  # Assign dict to the JSONField

    #             summary = report_data.get("summary", {})
    #             total = summary.get("total", 0)
    #             collected = summary.get("collected", 0)
    #             passed = summary.get("passed", 0)

    #             if collected == total:
    #                 if passed == total:
    #                     db_object.status = "success"
    #                 elif passed >= (total / 2):
    #                     db_object.status = "partial"
    #                 else:
    #                     db_object.status = "failed"
    #             db_object.save()

    #     except (FileNotFoundError, json.JSONDecodeError) as e:
    #         db_object.report = {"error": str(e)}  # Fallback or logging
    #         db_object.save()
    #     return True
    # except subprocess.CalledProcessError as e:
    #     db_object.status = "failed"
    #     db_object.save()
    #     return False
    try:
        if get_platform() == "windows":
            # On Windows, use shell=False for better control
            process = subprocess.Popen(
                command,
                env=env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        else:
        # Join command into a string for shell=True
            cmd_str = " ".join(command)

            # Run subprocess and stream output
            process = subprocess.Popen(
                cmd_str,
                env=env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

        for line in process.stdout:
            print(line, end="")  # Stream output live

        exit_code = process.wait()
        print("Subprocess ends")

        # ✅ Retry opening the report file to avoid file-lock issues
        for attempt in range(5):
            try:
                with open(report_file) as f:
                    report_data = json.load(f)
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(0.3)
        else:
            raise RuntimeError("Report file could not be opened after retries.")
        
        if db_object.platform == "linux_windows":
            current_key = "linux" if get_platform() != "windows" else "windows"

            db_object.refresh_from_db()
            # Initialize report if not set
            if not db_object.report:
                db_object.report = {}

            # Save current platform report
            db_object.report[current_key] = report_data
            db_object.save()  # Save after adding partial data

            # 🔄 Wait for other platform report to appear
            other_key = "windows" if current_key == "linux" else "linux"
            retries = 10  # total wait = retries * sleep_time
            sleep_time = 10  # seconds

            for attempt in range(retries):
                db_object.refresh_from_db()  # reload latest DB state
                if other_key in db_object.report:
                    print(f"{other_key} report found after {attempt+1} retries.")
                    break
                print(f"Waiting for {other_key} report... attempt {attempt+1}/{retries}")
                time.sleep(sleep_time)
            else:
                # Timeout waiting for other report
                print(f"⚠️ Timeout waiting for {other_key} report.")
                db_object.status = "partial"
                db_object.save()
                return  # Exit early, will update once the other side finishes
            
            # if get_platform == "linux" or get_platform =="macos" :
            # ✅ Both reports now exist, aggregate results
            linux_summary = db_object.report.get("linux", {}).get("summary", {})
            windows_summary = db_object.report.get("windows", {}).get("summary", {})

            total = linux_summary.get("total", 0) + windows_summary.get("total", 0)
            collected = linux_summary.get("collected", 0) + windows_summary.get("collected", 0)
            passed = linux_summary.get("passed", 0) + windows_summary.get("passed", 0)

            if collected == total:
                if passed == total:
                    db_object.status = "success"
                elif passed >= (total / 2):
                    db_object.status = "partial"
                else:
                    db_object.status = "failed"
            else:
                db_object.status = "failed"

            db_object.save()

        
        # # Try loading the report if process ran successfully
        # try:
        #     with open(report_file) as f:
        #         report_data = json.load(f)
        #         db_object.report = report_data

        #         # Parse summary and set status
        #         summary = report_data.get("summary", {})
        #         total = summary.get("total", 0)
        #         collected = summary.get("collected", 0)
        #         passed = summary.get("passed", 0)

        #         if collected == total:
        #             if passed == total:
        #                 db_object.status = "success"
        #             elif passed >= (total / 2):
        #                 db_object.status = "partial"
        #             else:
        #                 db_object.status = "failed"
        #         else:
        #             db_object.status = "failed"  # fallback

        #         db_object.save()

        #         os.remove(report_file)

        # except (FileNotFoundError, json.JSONDecodeError) as e:
        #     db_object.report = {"error": str(e)}
        #     db_object.status = "failed"
        #     db_object.save()
        else:
            # 🌐 Single platform (linux OR windows)
            db_object.report = report_data

            # 📊 Parse test summary and update status
            summary = report_data.get("summary", {})
            total = summary.get("total", 0)
            collected = summary.get("collected", 0)
            passed = summary.get("passed", 0)

            if collected == total:
                if passed == total:
                    db_object.status = "success"
                elif passed >= (total / 2):
                    db_object.status = "partial"
                else:
                    db_object.status = "failed"
            else:
                db_object.status = "failed"

            # Save the updated DB object
            db_object.save()
        # db_object.report = report_data

        # # 📊 Parse test summary and update status
        # summary = report_data.get("summary", {})
        # total = summary.get("total", 0)
        # collected = summary.get("collected", 0)
        # passed = summary.get("passed", 0)

        # if collected == total:
        #     if passed == total:
        #         db_object.status = "success"
        #     elif passed >= (total / 2):
        #         db_object.status = "partial"
        #     else:
        #         db_object.status = "failed"
        # else:
        #     db_object.status = "failed"

        # db_object.save()

        # 🧹 Cleanup
        if os.path.exists(report_file):
            os.remove(report_file)

        pipeline_file = f"config/pipeline_mode_bool_{timestamp}.txt"
        if os.path.exists(pipeline_file):
            os.remove(pipeline_file)


        return True

    except Exception as e:
        print(f"Error running test: {e}")
        db_object.status = "failed"
        db_object.save()
        pipeline_file = f"config/pipeline_mode_bool_{timestamp}.txt"
        if os.path.exists(pipeline_file):
            try:
                os.remove(pipeline_file)
            except Exception:
                pass
        # os.remove(f"config/pipeline_mode_bool_{timestamp}.txt")
        return False
    
def validate_api_key(key: str) -> bool:
    """Validate API key by attempting to list indexes via VectorX."""
    if not key:
        return False

    try:
        vx = VectorX(token=key)
        vx.list_indexes()
        return True
    except Exception as e:
        # Optionally log the error
        print(f"API key validation failed: {e}")
        return False
    


def update_library_in_requirements(library: str, version: str, requirements_file: str = "requirements.txt") -> bool:
    """
    Update the version of an existing library in requirements.txt.

    Assumes the library is already present in the file.

    Args:
        library (str): The library name (e.g., "vecx")
        version (str): The new version
        requirements_file (str): Path to requirements.txt

    Returns:
        bool: True if updated successfully
    """
    requirements_path = Path(requirements_file)
    if not requirements_path.exists():
        raise FileNotFoundError(f"{requirements_file} not found!")

    with open(requirements_file, "r") as f:
        lines = f.readlines()

    # Replace the version for the given library
    new_lines = [
        re.sub(rf"^{library}==.*", f"{library}=={version}", line, flags=re.IGNORECASE)
        if line.strip().lower().startswith(library.lower())
        else line
        for line in lines
    ]

    with open(requirements_file, "w") as f:
        f.writelines(new_lines)

    return True


def install_and_update_library(db_object: LibraryUpdateRequest, requirements_file: str = "requirements.txt") -> dict:
    """
    Install a library with a specific version and update requirements.txt if successful.
    """
    library = db_object.library
    new_version = db_object.update_version
    current_version = db_object.current_version
    package_spec = f"{library}=={new_version}"
    command = ["pip", "install", package_spec]

    install_result = subprocess.run(command, capture_output=True, text=True)

    pass_status = 0

    current_key = "linux" if get_platform() != "windows" else "windows"

    if install_result.returncode != 0:
        # New version installation failed immediately
        db_object.status = "failed"
        rollback_package_spec = f"{library}=={current_version}"
        rollback_command = ["pip", "install", rollback_package_spec]
        rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)
        db_object.refresh_from_db()
        # Initialize report if not set
        if not db_object.message:
            db_object.message = {}
            db_object.save()
        db_object.message[current_key] = {
            "status": "failed",
            "rollback": "true" if rollback_result.returncode == 0 else "false",
            "error": install_result.stderr.strip(),
            "rollback_output": rollback_result.stdout.strip() if rollback_result.returncode == 0 else rollback_result.stderr.strip()
        }
        db_object.save()
        return False
    
    # Step 2: Run pip check to detect hidden dependency issues
    pip_check_command = ["pip", "check"]
    check_result = subprocess.run(pip_check_command, capture_output=True, text=True)
    if check_result.returncode != 0:
        # New version installed, but caused dependency conflict — rollback
        rollback_package_spec = f"{library}=={current_version}"
        rollback_command = ["pip", "install", rollback_package_spec]
        rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)
        db_object.refresh_from_db()
        # Initialize report if not set
        if not db_object.message:
            db_object.message = {}
            db_object.save()
        db_object.message[current_key] =  {
            "status": "failed",
            "rollback": "true" if rollback_result.returncode == 0 else "false",
            "error": check_result.stderr.strip(),
            "rollback_output": rollback_result.stdout.strip() if rollback_result.returncode == 0 else rollback_result.stderr.strip()
        }
        db_object.save()
        return False

    db_object.refresh_from_db()
    # Initialize report if not set
    if not db_object.message:
        db_object.message = {}
        db_object.save()
    pass_status = 1
    db_object.message[current_key] =  {
            "status": "passed",
            "rollback": "false"
        }
    db_object.save()
    
    # 🔄 Wait for other platform message to appear
    other_key = "windows" if current_key == "linux" else "linux"
    retries = 10  # total wait = retries * sleep_time
    sleep_time = 10  # seconds

    for attempt in range(retries):
        db_object.refresh_from_db()  # reload latest DB state
        if other_key in db_object.message:
            print(f"{other_key} message found after {attempt+1} retries.")
            break
        print(f"Waiting for {other_key} message... attempt {attempt+1}/{retries}")
        time.sleep(sleep_time)
    else:
        # Timeout waiting for other report
        print(f"⚠️ Timeout waiting for {other_key} message.")
        pass_status = 0
        db_object.status = "failed"
        db_object.save()

        rollback_package_spec = f"{library}=={current_version}"
        rollback_command = ["pip", "install", rollback_package_spec]
        rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)

        if rollback_result.returncode == 0:
            db_object.message[current_key]["rollback"] = "true"
            db_object.message[current_key]["rollback_output"] = rollback_result.stdout.strip()
        else:
            db_object.message[current_key]["rollback"] = "false"
            db_object.message[current_key]["rollback_output"] = rollback_result.stderr.strip()
        db_object.save()
        return False
    
    
    if pass_status == 1:
        if db_object.message[other_key]["status"] == "failed":
            rollback_package_spec = f"{library}=={current_version}"
            rollback_command = ["pip", "install", rollback_package_spec]
            rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)

            if rollback_result.returncode == 0:
                db_object.message[current_key]["rollback"] = "true"
                db_object.message[current_key]["rollback_output"] = rollback_result.stdout.strip()
            else:
                db_object.message[current_key]["rollback"] = "false"
                db_object.message[current_key]["rollback_output"] = rollback_result.stderr.strip()
            db_object.save()
        
        else:
            db_object.status = "success"
            db_object.save()

    # # Installation succeeded, now update requirements.txt
    # try:
    #     update_library_in_requirements(library, new_version, requirements_file)
    #     return True
    
    # except Exception as e:
    #     db_object.status = "failed"
    #     db_object.message[current_key]["error"] = f"Installed {package_spec} but failed to update requirements.txt: {e}"
    #     db_object.save()

    #     rollback_package_spec = f"{library}=={current_version}"
    #     rollback_command = ["pip", "install", rollback_package_spec]
    #     rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)

    #     if rollback_result.returncode == 0:
    #         db_object.message[current_key]["rollback"] = "true"
    #         db_object.message[current_key]["rollback_output"] = rollback_result.stdout.strip()
    #     else:
    #         db_object.message[current_key]["rollback"] = "false"
    #         db_object.message[current_key]["rollback_output"] = rollback_result.stderr.strip()

    #     db_object.save()

        return False



# def get_library_version(lib_name):
#     try:
#         with open('requirements.txt') as f:
#             for line in f:
#                 if line.lower().startswith(lib_name.lower() + '=='):
#                     return line.strip().split('==')[1]
#     except Exception:
#         return "Unknown"
#     return "Not Found"
# def get_library_version(lib_name):
#     try:
#         return pkg_resources.get_distribution(lib_name).version
#     except pkg_resources.DistributionNotFound:
#         return "Not installed"
def get_library_version(lib_name):
    try:
        return version(lib_name)
    except PackageNotFoundError:
        return "Not installed"




