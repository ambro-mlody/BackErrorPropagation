import numpy as np

from neuralNetwork import NeuralNetwork
import pygame as py
import sys
import matplotlib.pyplot as plt
from examples import Examples, Point


def preprocess(x, xmin, xmax):
    x = np.array(x)
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    return 0.1 + (x - xmin) / (xmax - xmin) * 0.8


def postprocess(x, xmin, xmax):
    x = np.array(x)
    return xmin + (x - 0.1) * (xmax - xmin) / 0.8


py.init()
size = (400, 500)
window = py.display.set_mode(size)
py.display.set_caption("Robot")

arm_length = 150
ex = Examples(Point(0, size[1] / 2), arm_length)
training = ex.generate(1000)
point = Point(0, 0)
nn = NeuralNetwork([2, 10, 10, 2], 100, 0.01)
nn.train(preprocess(training[0], min(training[0]), max(training[0])), preprocess(training[1], min(training[1]), max(training[1])))
for e in nn.errors:
    print(e)

while True:
    window.fill((60, 60, 60))
    for ev in py.event.get():
        if ev.type == py.QUIT:
            py.quit()
            sys.exit(0)
        if py.mouse.get_pressed()[0]:
            pos = py.mouse.get_pos()
            point = Point(pos[0], pos[1])
            out = nn.forward(preprocess([point.x, point.y], 0, size[1]))
            #out = postprocess(out, min(out), max(out))
            print(out)
    point.draw(window)
    py.draw.circle(window, "green", (0, size[1] / 2), 8)
    py.display.update()

