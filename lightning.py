import time
import subprocess

from smbus2 import SMBus, i2c_msg


class Lightning:
    def __init__(self) -> None:
        subprocess.Popen(["gpioset", "gpiochip2", "19=1"])
        subprocess.Popen(["gpioset", "gpiochip2", "16=1"])
        self.bus_number = 7
        self.device_address = 0x12
        self.LED_array = [
            [11, 70, 0, 0, 0, 0, 0, 0, 0, 0, 255, 88],
            [11, 70, 7, 0, 7, 0, 7, 0, 7, 0, 1, 88],
            [11, 70, 7, 0, 7, 0, 7, 0, 7, 0, 2, 88],
            [11, 70, 7, 0, 7, 0, 7, 0, 7, 0, 4, 88],
            [11, 70, 7, 0, 7, 0, 7, 0, 7, 0, 8, 88],
            [11, 70, 0, 255, 0, 255, 1, 255, 1, 255, 64, 88],
            [11, 70, 0, 255, 0, 255, 1, 255, 1, 255, 16, 88],
            [11, 70, 7, 255, 7, 255, 10, 255, 15, 255, 128, 88],
            [11, 70, 7, 255, 7, 255, 10, 255, 15, 255, 32, 88],
        ]
        self.sortBite = [0x03, 0x01, 0x01, 0x02]

    def start(self) -> None:
        with SMBus(self.bus_number) as bus:
            for data in self.LED_array:
                write = i2c_msg.write(self.device_address, data)
                time.sleep(0.1)
                bus.i2c_rdwr(write)

    def stop(self) -> None:
        subprocess.Popen(["gpioset", "gpiochip2", "19=0"])
        subprocess.Popen(["gpioset", "gpiochip2", "16=0"])

    def update(self) -> None:
        pass
