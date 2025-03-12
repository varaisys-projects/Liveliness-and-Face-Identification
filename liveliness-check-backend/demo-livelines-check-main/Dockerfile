# init a base image (Alpine is small Linux distro)
FROM python:3.10
# define the present working directory
RUN mkdir -p /app
COPY . /app
# run pip to install the dependencies of the flask app
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx
RUN pip install opencv-python==4.8.1.78

ENV REMOTE_SERVER_PORT 3004 
ENV TOLERANCE_OF_MODEL 0.60
ENV STARTING_TIME_LIMIT_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME 10
ENV FRAMES_REQUIRED_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME_TO_LIVELINESS_TEST 35
ENV TIME_LIMIT_FOR_TOTAL_BLINK_DETECTION 35
ENV TIME_LIMIT_FOR_EACH_ACTION_IN_FACE_MOVEMENT_FUNCTION 50 

WORKDIR /app
# define the command to start the container
CMD ["python3","server_final_multiple_clients.py"]
