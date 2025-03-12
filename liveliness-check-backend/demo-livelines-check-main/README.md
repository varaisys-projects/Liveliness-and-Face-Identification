Liveliness Detection API.
 
Initially, we carry out face detection to make sure that the face is within the frame. Following that, we give instructions for the user to follow and check if they are present in real-time. Finally, we confirm the liveness by conducting face recognition.

Development stage setup for running the code locally  

1. clone the git repository into the projects folder on machine.

2. Within the projects folder, create another folder called 'venv'. This would have the virtual environments for the code.

3. To create a virtual environment for liveliness detection api, go to projects/venv folder. Open terminal and write the following commands: 
a. virtualenv liveliness_venv
(This would create a virtual env wherein the dependencies will be installed)

4. Go back to the liveliness detection api project directory. Open terminal and write the following command to activate the virtual environment there:   
a.  . ../venv/liveliness_venv/bin/activate  
b. pip install -r requirements.txt

5. Create a .env file, if not present, add the following env variable values:  
REMOTE_SERVER_PORT="3004"  
TOLERANCE_OF_MODEL=  
STARTING_TIME_LIMIT_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME=  
FRAMES_REQUIRED_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME_TO_LIVELINESS_TEST=  
TIME_LIMIT_FOR_TOTAL_BLINK_DETECTION=  
TIME_LIMIT_FOR_EACH_ACTION_IN_FACE_MOVEMENT_FUNCTION=  

6. To run the api:   
a. python server_final_multiple_clients.py


Docker

$ docker build -t  liveliness_image .   
 
$ docker run -d -p 3004:3004 -v $(pwd)/data:/app/data IMAGE_ID 
