import xml.etree.ElementTree as et
import numpy as np
import math


class Tree:
    def __init__(self, path):
        self.tree = et.parse(path)
        self.coordinates = self.__unique_coordinates()

    def get_unique_coordinates(self):
        return self.coordinates

    def __unique_coordinates(self):
        coordinates = []

        for N in self.tree.getroot():
            coordinates.append(self.coordinate(N))

        return np.array(list(set(coordinates)))

    def find_shortest_coordinate(self, lat: float, long: float) -> np.array:
        dot = np.array([lat, long]) * math.pi / 180;
        coordinates = self.coordinates * math.pi / 180

        cos_l1 = np.cos(coordinates[:, 0])
        cos_l2 = np.cos(dot[0])
        sin_l1 = np.sin(coordinates[:, 0])
        sin_l2 = np.sin(dot[0])
        delta = coordinates[:, 1] - dot[1]

        y = np.sqrt(np.square(cos_l2 * np.sin(delta)) + np.square(cos_l1 * sin_l2 - sin_l1 * cos_l2 * np.cos(delta)))
        x = sin_l1 * sin_l2 + cos_l1 * cos_l2 * np.cos(delta)
        distances = np.arctan2(y, x) * 6372795

        return self.coordinates[distances.argmin()]

    @staticmethod
    def coordinate(element) -> tuple:
        return float(element.attrib['Lat']), float(element.attrib['Long'])
