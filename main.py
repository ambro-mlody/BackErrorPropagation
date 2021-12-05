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
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    return xmin + (x - 0.1) * (xmax - xmin) / 0.8


py.init()
size = (400, 500)
window = py.display.set_mode(size)
py.display.set_caption("Robot")

arm_length = 100
center = Point(0, size[1] / 2)
ex = Examples(Point(center.x, center.y), arm_length)
training = ex.generate(1000)

point = Point(0, 0)
tempoint = Point(0, 0)
finalpoint = Point(0, 0)
nn = NeuralNetwork([2, 32, 2], 1000, 0.01)
pmin = (0, 0)
amin = 0
amax = np.pi
training_xx = []
training_xy = []
training_x = []
training_y = []
for t in training[0]:
    t_xx = preprocess(t[0], pmin[0], size[0])
    t_xy = preprocess(t[1], pmin[1], size[1])
    training_x.append([t_xx, t_xy])
for t in training[1]:
    t_yx = preprocess(t[0], amin, amax)
    t_yy = preprocess(t[1], amin, amax)
    training_y.append([t_yx, t_yy])
#nn.train(training_x, training_y)
#for e in nn.errors:
    #print(e)


def switch(key):
    if key == py.K_l:
        nn.train(training_x, training_y)
    elif key == py.K_p:
        plt.plot(nn.errors)
        plt.show()


while True:
    window.fill((60, 60, 60))
    for ev in py.event.get():
        if ev.type == py.QUIT:
            py.quit()
            sys.exit(0)
        if py.mouse.get_pressed()[0]:
            pos = py.mouse.get_pos()
            point = Point(pos[0], pos[1])
            p_x = preprocess(point.x, pmin[0], size[0])
            p_y = preprocess(point.y, pmin[0], size[1])
            out = nn.forward([p_x, p_y])
            out = (postprocess(out[0], amin, amax), postprocess(out[1], amin, amax))
            #print(out)
            tempoint = ex.translate(center, out[0])
            finalpoint = ex.translate(tempoint, np.pi - out[1] + out[0])
        if ev.type == py.KEYDOWN:
            switch(ev.key)
    tempoint.draw(window)
    finalpoint.draw(window)
    #for t in training[0]:
    #    tp = Point(t[0], t[1])
    #    tp.draw(window)
    py.draw.line(window, "blue", center.tup(), tempoint.tup())
    py.draw.line(window, "yellow", tempoint.tup(), finalpoint.tup())
    point.draw(window)
    py.draw.circle(window, "green", (0, size[1] / 2), 8)
    py.display.update()
