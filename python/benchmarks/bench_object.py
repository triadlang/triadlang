import math

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
total = 0.0
for i in range(5000):
    p1 = Point(float(i), float(i * 2))
    p2 = Point(float(i + 1), float(i * 2 + 1))
    total += p1.distance(p2)
print(f'object total = {total}')