import Rpi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.cleanup()

GPIO.setup(11, GPIO.OUT)
GPIO.output(11, True)