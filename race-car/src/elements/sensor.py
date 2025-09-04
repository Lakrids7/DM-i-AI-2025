# ..elements/sensor.py

import pygame
from ..mathematics.vector import Vector
from typing import List, Optional
from ..mathematics.collision import get_intersection_point, get_lines_of_rectangle
from .car import Car

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Sensor:
    def __init__(self, car: Car, angle: float, name: str, state):
        """
        Initialize a Sensor object.
        """
        self.car = car
        self.name = name
        self.degrees = angle
        self.reading = None

        self.sensor_width = 2
        # --- CHANGE #1: Rename this attribute from self.sensor_color to self.color ---
        self.color = (255, 0, 0)  # Red (default color)
        self.sensor_strength = 1000

        # ... rest of the __init__ method is unchanged ...
        vector = Vector(0, -self.sensor_strength).rotate(self.degrees)
        self.beam_start = (0, 0)
        self.beam_end = (vector.x, vector.y)
        self.text_position = (vector.x * 0.3, vector.y * 0.3)
        self.text = ""
        self.state = state

    def update(self):

        visible = self.state.sensors_enabled
        car_rect = self.car.get_bounds()
        car_center = Vector(car_rect.centerx, car_rect.centery)
        self.beam_start = (car_center.x, car_center.y)
        sensor_beam_end = car_center.add(Vector(0, -self.sensor_strength).rotate(self.degrees))
        self.beam_end = (sensor_beam_end.x, sensor_beam_end.y)
        self.reading = None
        sensor_line = Line(car_center, sensor_beam_end)
        min_reading = None

        for car in self.state.cars:
            if car == self.state.ego:
                continue
            bounds = car.get_bounds()
            reading = self.get_sensor_reading_for_bounding_box(bounds, sensor_line, car_center)
            if reading is not None and 0 <= reading <= self.sensor_strength:
                if min_reading is None or reading < min_reading:
                    min_reading = reading
        for wall in self.state.road.walls:
            bounds = wall.get_bounds()
            reading = self.get_sensor_reading_for_bounding_box(bounds, sensor_line, car_center)
            if reading is not None and 0 <= reading <= self.sensor_strength:
                if min_reading is None or reading < min_reading:
                    min_reading = reading
        self.reading = min_reading

        if self.reading is not None:
            self.text = f"{self.reading:.2f}"
        else:
            self.text = ""

    def get_sensor_reading_for_bounding_box(self, bb: pygame.Rect, sensor_line: dict, car_center: Vector) -> Optional[float]:
        # ... this method is unchanged ...
        lines = get_lines_of_rectangle(bb)
        min_distance = None
        for line in lines:
            intersection = get_intersection_point(sensor_line, line)
            if intersection:
                distance = car_center.distance(intersection)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance

    def draw(self, surface: pygame.Surface):
        """
        Draw the sensor beam and text on the given surface.
        """
        if self.state.sensors_enabled:
            # --- CHANGE #2: Use self.color here instead of self.sensor_color ---
            pygame.draw.line(surface, self.color, self.beam_start, self.beam_end, self.sensor_width)

            # ... rest of the draw method is unchanged ...
            if self.text:
                font = pygame.font.SysFont("monospace", 16)
                text_surface = font.render(self.text, True, (255, 255, 255))
                text_x = self.beam_start[0] + 0.3 * (self.beam_end[0] - self.beam_start[0])
                text_y = self.beam_start[1] + 0.3 * (self.beam_end[1] - self.beam_start[1])
                surface.blit(text_surface, (text_x, text_y))