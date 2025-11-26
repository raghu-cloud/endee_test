from django.contrib import admin
from django import forms
from django.shortcuts import redirect
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.template.loader import render_to_string
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.db import transaction
from .utils import validate_api_key, run_test, get_library_version
from .models import TestRun, UserAPIKey, get_platform, LibraryUpdateRequest
from datetime import datetime
import subprocess
import json
import sys
import os
import requests
from dotenv import load_dotenv

load_dotenv()
WINDOWS_SERVER_IP = os.getenv("WINDOWS_SERVER_IP")

# Register your models here.

class TestRunAdminForm(forms.ModelForm):
    use_fallback = forms.BooleanField(
        label="Use fallback API key",
        required=False,
        initial=True,
        help_text="Uncheck to enter your own API key"
    )
    custom_api_key = forms.CharField(
        label="Your API Token",
        required=False,
        widget=forms.PasswordInput(render_value=True),
        help_text="Leave blank to use fallback key"
    )
    test_type = forms.ChoiceField(
        choices=TestRun.TEST_CHOICES,
        label="Test Type",
        help_text="Select which test to execute"
    )
    platform = forms.ChoiceField(
        choices=TestRun.PLATFORM_CHOICES,  # Assuming you have PLATFORM_CHOICES in TestRun
        label="Platform",
        help_text="Select the platform to run the test on"
    )

    class Meta:
        model = TestRun
        fields = ['test_type', 'platform', 'use_fallback', 'custom_api_key']

class LibraryUpdateAdminForm(forms.ModelForm):
    library = forms.ChoiceField(
        choices=LibraryUpdateRequest.LIBRARY_CHOICES,
        label="Library",
        help_text="Select the library whose version is to be updated"
    )
    current_version = forms.CharField(
        label="Current Version",
        required=False,
        disabled=True  # makes it read-only
    )
    update_version = forms.CharField(
        label="Update Version",
        help_text="Enter the new version you want to update to"
    )

    class Meta:
        model = LibraryUpdateRequest  # Replace with your model
        fields = ['library', 'current_version', 'update_version', 'status', 'message']

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    #     print("Form __init__ called")
    #     print("self.data:", self.data)
    #     print("self.instance:", self.instance)
    #     if 'library' in self.data:
    #         lib = self.data.get('library')
    #         print("oooo",get_library_version(lib))
    #         self.fields['current_version'].initial = get_library_version(lib)
    #     elif self.instance and self.instance.library:
    #         print("library from instance:", self.instance.library)
    #         self.fields['current_version'].initial = get_library_version(self.instance.library)
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    #     lib = None
    #     if self.data.get('library'):
    #         lib = self.data.get('library')
    #     elif self.instance and self.instance.library:
    #         lib = self.instance.library

    #     if lib:
    #         version = get_library_version(lib)
    #         self.fields['current_version'].initial = version
    #         self.initial['current_version'] = version  # ✅ ensures it's passed into the form cleaned_data


@admin.register(TestRun)
class TestRunAdmin(admin.ModelAdmin):
    form = TestRunAdminForm
    list_display = ('id', 'user', 'get_test_type_display', 'platform', 'status', 'timestamp')
    list_filter = ('status', 'test_type')
    # readonly_fields = ('status','formatted_report', 'timestamp', 'vectorx_version')
    exclude = ('report',)  # Exclude the raw report field

    def get_readonly_fields(self, request, obj=None):
        base_fields = ['status', 'formatted_report', 'timestamp', 'vectorx_version']
        if obj:  # This means we are editing an existing object
            return base_fields + ['test_type', 'platform']
        return base_fields
    
    # def get_fields(self, request, obj=None):
    #     """Override to control field order and visibility"""
    #     if obj:  # Editing existing object
    #         return ('test_type', 'status', 'formatted_report', 'timestamp', 'vectorx_version')
    #     else:  # Creating new object
    #         return ('test_type', 'use_fallback', 'custom_api_key')

    def get_changeform_initial_data(self, request):
        initial = super().get_changeform_initial_data(request)
        # Pre-populate with user's existing API key if available
        try:
            api_key = UserAPIKey.objects.get(user=request.user).key
            initial.update({
                'use_fallback': False,
                'custom_api_key': api_key
            })
        except UserAPIKey.DoesNotExist:
            # pass
            initial['platform'] = 'linux'
        return initial
    
    def save_model(self, request, obj, form, changed):
        # Handle API key first
        use_fallback = form.cleaned_data.get('use_fallback', True)
        custom_key = form.cleaned_data.get('custom_api_key', '').strip()
        
        if not use_fallback:
            if custom_key:
                if not validate_api_key(custom_key):
                    messages.error(request, "Invalid API key - please re-enter or use fallback")
                    return HttpResponseRedirect(request.path)
                
                # Save/update the API key
                UserAPIKey.objects.update_or_create(
                    user=request.user,
                    defaults={'key': custom_key}
                )
                obj.api_key_reference = UserAPIKey.objects.get(user=request.user)
            else:
                messages.error(request, "Please either enter an API key or check 'Use fallback'")
                return HttpResponseRedirect(request.path)
        # Save the chosen platform to obj
        obj.platform = form.cleaned_data.get('platform')

        # Set the user and initial status
        obj.user = request.user
        obj.status = 'pending'
        
        super().save_model(request, obj, form, changed)
        
         # Add success message and run test in background using threading
        messages.success(request, f"Test run #{obj.id} has been created and will start shortly. You can refresh this page to see the status.")
        
        # Run the test in a separate thread to avoid blocking the HTTP response
        # import threading
        # thread = threading.Thread(target=run_test, args=(obj,))
        # thread.daemon = True  # Dies when main thread dies
        # thread.start()
        # Prepare subprocess command
        # if get_platform() == "windows":
        #     command = [sys.executable, 'manage.py', 'run_test_job', str(obj.id)]
        # else:
        #     command = ['python', 'manage.py', 'run_test_job', str(obj.id)]
        
        
        # # Optionally pass API key only if it exists
        # if obj.api_key:
        #     command += ['--api_key', obj.api_key]

        # subprocess.Popen(command)
        if obj.platform == "windows":
            # Call API on Windows server
            try:
                response = requests.post(
                    f"http://{WINDOWS_SERVER_IP}:8000/run-tests/",
                    json={"test_run_id": obj.id},
                    headers={"X-API-KEY": obj.api_key} if obj.api_key else {},
                    timeout=5  # Short timeout for API response
                )
                if response.status_code == 200:
                    messages.success(request, f"Triggered test run #{obj.id} on Windows server.")
                else:
                    messages.error(request, f"Failed to trigger test on Windows server: {response.status_code} {response.text}")
            except requests.RequestException as e:
                messages.error(request, f"Error contacting Windows server: {e}")

        elif obj.platform == "linux":
            # Directly run subprocess for Linux
            try:
                command = ['python', 'manage.py', 'run_test_job', str(obj.id)]
                subprocess.Popen(command)
                if obj.api_key:
                    command += ['--api_key', obj.api_key]
                messages.success(request, f"Test run #{obj.id} started on Linux server.")
            except Exception as e:
                messages.error(request, f"Failed to start test on Linux server: {e}")

        elif obj.platform == "linux_windows":
            # Run test on Linux first
            linux_command = [sys.executable, 'manage.py', 'run_test_job', str(obj.id)]
            if obj.api_key:
                linux_command += ['--api_key', obj.api_key]
            subprocess.Popen(linux_command)

            # Then trigger Windows server
            try:
                response = requests.post(
                    f"http://{WINDOWS_SERVER_IP}:8000/run-tests/",
                    json={"test_run_id": obj.id},
                    headers={"X-API-KEY": obj.api_key} if obj.api_key else {},
                    timeout=5
                )
                if response.status_code == 200:
                    messages.success(request, f"Triggered test run #{obj.id} on both Linux and Windows servers.")
                else:
                    messages.error(request, f"Linux test started, but Windows API failed: {response.text}")
            except Exception as e:
                messages.error(request, f"Linux test started, but error calling Windows server: {e}")

        # messages.success(request, f"Test run #{obj.id} has been created and will start shortly.")

    def formatted_report(self, obj):
        """Display the test report as formatted HTML instead of raw JSON"""
        if not obj.report:
            return "No report available"
        
        try:
            # Parse the JSON report
            report_data = obj.report
            
            if obj.platform == "linux_windows":
                timestamp = report_data.get('linux', {}).get('created')
            else:
                timestamp = report_data.get('created')
            if timestamp:
                try:
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    report_data['created_readable'] = readable_time
                except (ValueError, TypeError):
                    report_data['created_readable'] = "Invalid Timestamp"
            else:
                report_data['created_readable'] = "N/A"


                # Extract and categorize tests
            if obj.platform == "linux_windows":
                linux_tests = report_data.get("linux", {}).get("tests", [])
                windows_tests = report_data.get("windows", {}).get("tests", [])
                for test in linux_tests:
                    setup_duration = test.get("setup", {}).get("duration", 0)
                    call_duration = test.get("call", {}).get("duration", 0)
                    teardown_duration = test.get("teardown", {}).get("duration", 0)
                    
                    # Add total_duration field
                    test["total_duration"] = setup_duration + call_duration + teardown_duration
                for test in windows_tests:
                    setup_duration = test.get("setup", {}).get("duration", 0)
                    call_duration = test.get("call", {}).get("duration", 0)
                    teardown_duration = test.get("teardown", {}).get("duration", 0)
                    
                    # Add total_duration field
                    test["total_duration"] = setup_duration + call_duration + teardown_duration

                linux_passed_tests = [test for test in linux_tests if test.get('outcome') == 'passed']
                linux_failed_tests = [test for test in linux_tests if test.get('outcome') == 'failed']
                linux_error_tests = [test for test in linux_tests if test.get('outcome') == 'error']
                linux_skipped_tests = [test for test in linux_tests if test.get('outcome') == 'skipped']

                windows_passed_tests = [test for test in windows_tests if test.get('outcome') == 'passed']
                windows_failed_tests = [test for test in windows_tests if test.get('outcome') == 'failed']
                windows_error_tests = [test for test in windows_tests if test.get('outcome') == 'error']
                windows_skipped_tests = [test for test in windows_tests if test.get('outcome') == 'skipped']

                context = {
                    'linux_passed_tests': linux_passed_tests,
                    'linux_failed_tests': linux_failed_tests,
                    'linux_error_tests': linux_error_tests,
                    'linux_skipped_tests': linux_skipped_tests,
                    'linux_total_tests': len(linux_tests),
                    'linux_passed_count': len(linux_passed_tests),
                    'linux_failed_count': len(linux_failed_tests),
                    'windows_passed_tests': windows_passed_tests,
                    'windows_failed_tests': windows_failed_tests,
                    'windows_error_tests': windows_error_tests,
                    'windows_skipped_tests': windows_skipped_tests,
                    'windows_total_tests': len(windows_tests),
                    'windows_passed_count': len(windows_passed_tests),
                    'windows_failed_count': len(windows_failed_tests),
                    'report': report_data
                }

                # Render the template
                html_content = render_to_string('admin/test_report_display_linux_windows.html', context)

            else:
                tests = report_data.get('tests', [])
                for test in tests:
                    setup_duration = test.get("setup", {}).get("duration", 0)
                    call_duration = test.get("call", {}).get("duration", 0)
                    teardown_duration = test.get("teardown", {}).get("duration", 0)

                    # Add total_duration field
                    test["total_duration"] = setup_duration + call_duration + teardown_duration
                passed_tests = [test for test in tests if test.get('outcome') == 'passed']
                failed_tests = [test for test in tests if test.get('outcome') == 'failed']
                error_tests = [test for test in tests if test.get('outcome') == 'error']
                skipped_tests = [test for test in tests if test.get('outcome') == 'skipped']
                
                # Prepare context for template
                context = {
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'error_tests': error_tests,
                    'skipped_tests': skipped_tests,
                    'total_tests': len(tests),
                    'passed_count': len(passed_tests),
                    'failed_count': len(failed_tests),
                    'report': report_data
                }
            
                # Render the template
                html_content = render_to_string('admin/test_report_display.html', context)

            return mark_safe(html_content)
        
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to raw JSON if parsing fails
            return format_html(
                '<div style="background: #f8d7da; padding: 10px; border-radius: 5px;">'
                '<strong>Error parsing report:</strong> {}<br><br>'
                '<details><summary>Raw JSON Data</summary><pre>{}</pre></details>'
                '</div>',
                str(e),
                obj.report
            )
        
    formatted_report.short_description = "Test Report"
            

    def get_fields(self, request, obj=None):
        if obj:  # Viewing an existing TestRun (after save)
            return ['test_type', 'status', 'platform','formatted_report', 'timestamp', 'vectorx_version']
        else:  # Creating a new TestRun (initial form)
            return ['test_type', 'platform', 'use_fallback', 'custom_api_key']


@admin.register(UserAPIKey)
class UserAPIKeyAdmin(admin.ModelAdmin):
    list_display = ('user', 'key_masked', 'last_updated')
    readonly_fields = ('last_updated',)
    exclude = ('_key_encrypted',)
    
    def key_masked(self, obj):
        return f"{obj.key[:4]}...{obj.key[-4:]}" if obj.key else ""
    key_masked.short_description = "API Key"


@admin.register(LibraryUpdateRequest)
class LibraryUpdateAdmin(admin.ModelAdmin):
    form = LibraryUpdateAdminForm
    list_display = ('id', 'user', 'library', 'update_version', 'current_version', 'status', 'timestamp')
    list_filter = ('status', 'library')
    # readonly_fields = ['current_version', 'status', 'message', 'timestamp']

    class Media:
        js = ('admin/js/library_version_fetch.js',)

    def get_readonly_fields(self, request, obj=None):
        base_fields = ['current_version', 'status', 'message', 'timestamp']
        if obj:  # This means we are editing an existing object
            return base_fields + ['library', 'update_version']
        return base_fields


    def save_model(self, request, obj, form, change):
        # Fetch the current version from your external source
        current_version = get_library_version(obj.library)

        if obj.update_version.strip() == current_version:
            messages.error(request, f"Version {current_version} is already the current version.")
            return HttpResponseRedirect(request.path)  # Prevent saving
        
        obj.user = request.user
        obj.current_version = current_version  # ✅ Ensure it's stored in DB
        obj.status = 'running'
        # If valid, allow saving as usual
        super().save_model(request, obj, form, change)

        linux_command = [sys.executable, 'manage.py', 'install_library', str(obj.id)]
        # Schedule subprocess only after commit
        transaction.on_commit(lambda: subprocess.Popen(linux_command))
        # subprocess.Popen(linux_command)

        try:
            response = requests.post(
                f"http://{WINDOWS_SERVER_IP}:8000/install-library/",
                json={"version_update_id": obj.id},
                timeout=5
            )
            if response.status_code == 200:
                messages.success(request, f"Triggered library version install #{obj.id} on both Linux and Windows servers.")
            else:
                messages.error(request, f"Library version install started on linux, but Windows API failed: {response.text}")
        except Exception as e:
            messages.error(request, f"Library version install started on Linux, but error calling Windows server: {e}")


    def get_fields(self, request, obj=None):
        if obj:  # Viewing an existing TestRun (after save)
            return ['status', 'library', 'current_version', 'update_version', 'message', 'timestamp']
        else:  # Creating a new TestRun (initial form)
            return ['library', 'current_version', 'update_version']










