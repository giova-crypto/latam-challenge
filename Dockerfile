# syntax=docker/dockerfile:1.2
FROM python:3.9-buster
# Set the working directory
WORKDIR /app

# copy necessary files for the container
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .

COPY . ./

# install dependencies
RUN pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt

# expose port
EXPOSE 8080

# finally run the application inside the container
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]