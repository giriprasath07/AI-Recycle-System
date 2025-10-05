import time
import gpiod
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as pca_servo

# -------- Servo Setup --------
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50
my_servo = pca_servo.Servo(pca.channels[0])

# -------- Stepper Setup --------
chip = gpiod.Chip('gpiochip4')
STEP_PIN = 21
DIR_PIN = 20
step_line = chip.get_line(STEP_PIN)
dir_line = chip.get_line(DIR_PIN)
step_line.request(consumer="step", type=gpiod.LINE_REQ_DIR_OUT)
dir_line.request(consumer="dir", type=gpiod.LINE_REQ_DIR_OUT)

def move_stepper(steps, direction, step_delay=0.003):
    dir_line.set_value(direction)
    time.sleep(0.00005)
    for _ in range(steps):
        step_line.set_value(1)
        time.sleep(0.00002)
        step_line.set_value(0)
        time.sleep(step_delay)

def flip_servo(angle):
    my_servo.angle = angle
    time.sleep(1)
    my_servo.angle = 125
    time.sleep(1)

# -------- Bin Actions --------
def general_bin():
    print("Routing to general bin")
    move_stepper(50, 0)
    time.sleep(1)
    flip_servo(70)
    move_stepper(50, 1)

def sharp_bin():
    print("Routing to sharp bin")
    move_stepper(50, 1)
    time.sleep(1)
    flip_servo(70)
    move_stepper(50, 0)
    
def pharamaceutical_bin():
    print("Routing to pharametic bin")
    move_stepper(50, 0)
    time.sleep(1)
    flip_servo(180)
    move_stepper(50, 1)
    
def hazard_bin():
    print("Routing to hazard bin")
    move_stepper(50, 1)
    time.sleep(1)
    flip_servo(180)
    move_stepper(50, 0)

def cleanup():
    step_line.release()
    dir_line.release()
    pca.deinit()
