# Use environment variable to set the python version
# Use a full Python image
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}

WORKDIR /usr/src/app
# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Upgrade pip and Install packages from requirements.txt
RUN pip install --upgrade pip && \
    # Install any needed packages specified in requirements.txt
    pip install --no-cache-dir -r requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "myapp/api/sentiment_predictor.py"]
