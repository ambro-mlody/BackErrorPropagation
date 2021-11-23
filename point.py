import pygame as py


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, window):
        py.draw.circle(window, "red", (self.x, self.y), 4)

    def tup(self):
        return self.x, self.y