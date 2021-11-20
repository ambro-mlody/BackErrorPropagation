import numpy as np
from point import Point


class Examples(object):
    def __init__(self, center, arm_length=1):
        self.arm_length = arm_length
        self.center = center  # Point(0, 0)
        self.input = []
        self.output = []

    def generate(self, number_of_examples):
        for _ in range(number_of_examples):
            point = self.generate_point()
            self.input.append([point.x, point.y])
        return self.input, self.output

    def generate_point(self):
        alpha = np.random.random() * np.pi
        beta = np.random.random() * np.pi
        self.output.append([alpha, beta])
        tempoint = self.translate(self.center, alpha)
        finalpoint = self.translate(tempoint, np.pi - beta + alpha)
        return finalpoint

    def translate(self, center, angle):
        return Point(center.x + self.arm_length * np.sin(angle), center.y - self.arm_length * np.cos(angle))
