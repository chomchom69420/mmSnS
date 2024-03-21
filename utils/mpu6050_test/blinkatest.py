import board
import busio

print("Hello blinka!")
i2c = busio.I2C(board.SCL, board.SDA)
print("I2C1 ok!")

print("done!")
