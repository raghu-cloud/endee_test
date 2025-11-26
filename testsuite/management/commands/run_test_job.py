from django.core.management.base import BaseCommand
from testsuite.models import TestRun
from testsuite.utils import run_test

class Command(BaseCommand):
    help = 'Run test job using ID and API key'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type = int)
        parser.add_argument('--api_key', type=str, help='Optional API key')

    def handle(self, *args, **options):
        job_id = options['job_id']
        api_key = options.get('api_key', None)

        try:
            test_job = TestRun.objects.get(id=job_id)
        except TestRun.DoesNotExist:
            self.stderr.write(f"TestJob with id {job_id} not found.")
            return
        
        self.stdout.write(f"Running test for test type code{test_job.test_type} with API token(key): {api_key}")

        run_test(test_job, api_key)