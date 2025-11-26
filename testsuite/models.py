from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import platform
import os
import re
import pkg_resources
from importlib.metadata import version, PackageNotFoundError

# Create your models here.
load_dotenv()

def get_fernet():
    FERNET_SECRET = os.getenv("FERNET_SECRET")
    return Fernet(FERNET_SECRET.encode())

def get_platform():
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    elif system.startswith("darwin"):
        return "macos"
    else:
        return "linux"

# def get_vecx_version_from_requirements():
#     try:
#         requirements_path = os.path.join(settings.BASE_DIR, 'requirements.txt')
        
#         with open(requirements_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 # Match patterns like: vecx==1.2.3, vecx>=1.2.3, vecx~=1.2.3
#                 if line.startswith('vecx'):
#                     # Extract version using regex
#                     match = re.search(r'vecx[>=~!]+([\w\.\-]+)', line)
#                     if match:
#                         return match.group(1)
#         return 'unknown'
#     except:
#         return 'unknown'
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
    

class UserAPIKey(models.Model):
    """Stores the user's last used API key persistently"""
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True
    )
    _key_encrypted = models.TextField(db_column='key')
    last_updated = models.DateTimeField(auto_now=True)

    @property
    def key(self):
        if self._key_encrypted:
            try:
                return get_fernet().decrypt(self._key_encrypted.encode()).decode()
            except Exception:
                return None
        return None
    
    @key.setter
    def key(self, value):
        if value:
            self._key_encrypted = get_fernet().encrypt(value.encode()).decode()
        else:
            self._key_encrypted = ''

    def __str__(self):
        return f"API Key for {self.user.username}"


class TestRun(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="The user who triggered this test run"
    )

    vectorx_version = models.CharField(
        max_length=50,
        # default=get_library_version("vecx"),  # Auto-populate from package
        help_text="Version of VectorX used for this test run"
    )
    
    PLATFORM_CHOICES = [
        ('linux', 'Linux Only'),
        ('windows', 'Windows Only'),
        ('linux_windows', 'Run on Linux & Windows'),
    ]

    TEST_CHOICES = [
        (1, "Python vecx Client"),
        (2, "Hybrid Index"), 
        (3, "VectorX LangChain Integration"),
        (4, "VectorX LlamaIndex Integration"),
        (5, "VectorX Crew AI Integration"),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending - Waiting to start'),
        ('running', 'Running - Currently executing'),
        ('success', 'Success - All tests passed'),
        ('partial', 'Partial - Some tests failed'), 
        ('failed', 'Failed - All tests failed'),
    ]

    platform = models.CharField(
        max_length=15,
        choices=PLATFORM_CHOICES,
        null=True, blank=True,
        help_text="The platform (OS) where this test was executed"
    )

    test_type = models.IntegerField(choices=TEST_CHOICES)
    pipeline_mode = models.BooleanField(default=False)
    report = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    timestamp = models.DateTimeField(auto_now_add=True)

    api_key_reference = models.ForeignKey(
        UserAPIKey,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Reference to the API key used"
    )

    @property
    def api_key(self):
        if self.api_key_reference:
            return self.api_key_reference.key
        return None


    def save(self, *args, **kwargs):
        # Set pipeline mode flag
        self.pipeline_mode = (self.test_type == 1)  # Changed from test_code to test_type
        
        # Handle API key reference if user exists
        if self.user and not self.api_key_reference:
            try:
                self.api_key_reference = UserAPIKey.objects.get(user=self.user)
            except UserAPIKey.DoesNotExist:
                pass

        if not self.platform:
            self.platform = get_platform()

        if not self.vectorx_version:
            self.vectorx_version = get_library_version("vecx")
        
        super().save(*args, **kwargs)

    def __str__(self):
        return f"TestRun by {self.user} - (Code {self.test_type}) on {self.platform}"
    

class LibraryUpdateRequest(models.Model):
    LIBRARY_CHOICES = [
        ("vecx", "vecx"),
        ("vecx-langchain", "vecx-langchain"),
        ("vecx-llamaindex", "vecx-llamaindex")
    ]

    STATUS_CHOICES = [
        ("running", "Running"),
        ("success", "Success"),
        ("failed", "Failed"),
    ]

    library = models.CharField(max_length=100)
    current_version = models.CharField(max_length=50, help_text="The current version of the library")
    update_version = models.CharField(max_length=50, help_text="The version for which library has to be updated")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="running")
    message = models.JSONField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.library}=={self.update_version} ({self.status})"
