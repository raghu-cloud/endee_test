from django.core.management.base import BaseCommand
from testsuite.models import LibraryUpdateRequest
from testsuite.utils import install_and_update_library

class Command(BaseCommand):
    help = 'Install library and update requirements'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type = int)

    def handle(self, *args, **options):
        job_id = options['job_id']

        try:
            test_job = LibraryUpdateRequest.objects.get(id=job_id)
        except LibraryUpdateRequest.DoesNotExist:
            self.stderr.write(f"TestJob with id {job_id} not found.")
            return
        
        self.stdout.write(f"Installing version=={test_job.update_version} for library {test_job.library}")

        install_and_update_library(test_job)