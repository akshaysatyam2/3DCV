import numpy as np

class SimpleMesh:
    def __init__(self):
        # a simple triangle
        self.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        self.faces = np.array([
            [0, 1, 2] # connects vertex 0, 1, and 2
        ])

mesh = SimpleMesh()
print(f"mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
