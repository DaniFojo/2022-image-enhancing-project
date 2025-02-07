# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
COPY . .

# Expose port
EXPOSE 80

# command to run on container start
ENTRYPOINT ["python", "/app/application/app.py"]
