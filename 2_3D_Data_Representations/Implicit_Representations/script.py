import numpy as np

def implicit_sphere(x, y, z, radius=1.0):
    """returns distance to surface of a sphere. negative means inside."""
    return np.sqrt(x**2 + y**2 + z**2) - radius

# check a few points
p1 = (0, 0, 0)
p2 = (2, 0, 0)
p3 = (0, 1, 0)

print(f"point {p1} dist: {implicit_sphere(*p1)}")
print(f"point {p2} dist: {implicit_sphere(*p2)}")
print(f"point {p3} dist: {implicit_sphere(*p3)}")
