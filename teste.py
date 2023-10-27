from djitellopy import Tello
import time

tello = Tello()
tello.connect()

bateria = tello.get_battery()
print("BATERIA:", bateria)

tello.takeoff()
time.sleep(1)

z = tello.get_height()

tello.send_control_command("command")

tello.curve_xyz_speed(80, -20, z, 80, 80, z, 30)

tello.land()
