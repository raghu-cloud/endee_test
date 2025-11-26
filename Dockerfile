# Use Python 3.12 base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Run Django with Gunicorn
# CMD ["gunicorn", "vectorx_testing_platform.wsgi:application", "--bind", "0.0.0.0:8000"]
CMD ["gunicorn", "vectorx_testing_platform.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--log-level", "info"]
