# antes del ref

import numpy as np

class QuadTree:
    def __init__(self, data, max_depth=10, min_samples=10):
        self.data = data
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.children = None
        self.depth = 0
        self.indices = np.arange(data.shape[0])

    def _split(self):
        # Código para dividir el nodo en cuatro
        pass

    def build(self):
        if self.children is None and self.depth < self.max_depth and self.indices.size > self.min_samples:
            self._split()
            for child in self.children:
                child.build()

    def query(self, point, radius):
        # Código para consultar el árbol
        pass
    
    #despues del ref
    
    import numpy as np

class QuadTreeNode:
    def __init__(self, data, max_depth=10, min_samples=10, depth=0, indices=None):
        self.data = data
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.children = None
        self.depth = depth
        self.indices = indices if indices is not None else np.arange(data.shape[0])

    def _split(self):
        # Código para dividir el nodo en cuatro
        pass

    def build(self):
        if self.children is None and self.depth < self.max_depth and self.indices.size > self.min_samples:
            self._split()
            for child in self.children:
                child.build()

    def query(self, point, radius):
        # Código para consultar el árbol
        pass

class QuadTree:
    def __init__(self, data, max_depth=10, min_samples=10):
        self.root = QuadTreeNode(data, max_depth, min_samples)

    def build(self):
        self.root.build()

    def query(self, point, radius):
        return self.root.query(point, radius)
    

'''El refactoring se ha realizado para mejorar la legibilidad y la estructura del código. Se ha creado una clase QuadTreeNode que representa un nodo del árbol, y la clase QuadTree ahora solo se encarga de crear el árbol y de proporcionar una interfaz para consultar el árbol. Esto hace que el código sea más fácil de entender y de mantener.

Además, se ha añadido un parámetro indices al constructor de QuadTreeNode, lo que permite crear nodos con un subconjunto específico de los datos. Esto puede ser útil para crear nodos hijos al dividir un nodo.

Finalmente, se ha eliminado el método build de la clase QuadTree, ya que ahora se llama al método build del nodo raíz. Esto hace que el código sea más conciso y más fácil de entender.'''