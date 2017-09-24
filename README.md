# carla.hal
Real Self-Driving Car

if you are experiencing an issue of the car not moving on launch in the current master branch a quick fix is to replace

sio = socketio.Server() on server.py

with 

eventlet.monkey_patch()
sio = socketio.Server(async_mode='eventlet')

