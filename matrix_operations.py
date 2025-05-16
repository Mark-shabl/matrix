class Matrix():
    def    __init__(self, matrix):
        if not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("Все строки в матрице должны иметь одинаковую длину.")
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows > 0 else 0

    def __str__(self):
        f = ""
        for i in self.matrix:
            f += str(i) + "\n"
        f = f.strip()
        return f

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы должны иметь одинаковые размеры для сложения.")

        result = [
            [self.matrix[i][j] + other.matrix[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы должны иметь одинаковые размеры для вычитания.")

        result = [
            [self.matrix[i][j] - other.matrix[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Умножение матрицы на число
            result = [[self.matrix[i][j] * other for j in range(self.cols)]
                      for i in range(self.rows)]
            return Matrix(result)
        elif isinstance(other, Matrix):
            # Умножение матрицы на матрицу
            if self.cols != other.rows:
                raise ValueError(
                    "Количество столбцов в первой матрице должно соответствовать количеству строк во второй матрице."
                )

            result = [
                [sum(self.matrix[i][k] * other.matrix[k][j] for k in range(self.cols))
                 for j in range(other.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        else:
            raise TypeError("Неподдерживаемый тип операнда для умножения.")

    def determinant(self, mat=None):
        if mat is None:
            mat = self.matrix
            if self.rows != self.cols:
                raise ValueError("Для вычисления определителя матрица должна быть квадратной.")

        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        sign = 1
        for i in range(n):
            minor = [row[:i] + row[i + 1:] for row in mat[1:]]
            det += sign * mat[0][i] * self.determinant(minor)
            sign = -sign
        return det

    def is_connected(self):
        if self.rows != self.cols:
            raise ValueError("Для проверки связности матрица должна быть квадратной.")

        # Преобразуем матрицу в список смежности
        adjacency_list = []
        for i in range(self.rows):
            neighbors = []
            for j in range(self.cols):
                if self.matrix[i][j] != 0:
                    neighbors.append(j)
            adjacency_list.append(neighbors)

        visited = set()
        stack = [0]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for neighbor in adjacency_list[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return len(visited) == self.rows

    def kruskal(self):
        if self.rows != self.cols:
            raise ValueError("Для алгоритма Крускала матрица должна быть квадратной.")

        # Создаем список всех ребер с их весами
        edges = []
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if self.matrix[i][j] != 0:
                    edges.append((self.matrix[i][j], i, j))

        # Сортируем ребра по весу
        edges.sort()

        parent = [i for i in range(self.rows)]

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        result = []
        for weight, u, v in edges:
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                result.append((u, v, weight))
                parent[v_root] = u_root

        # Создаем матрицу минимального остовного дерева
        min_span_tree = [[0 for _ in range(self.rows)] for _ in range(self.rows)]
        for u, v, weight in result:
            min_span_tree[u][v] = weight
            min_span_tree[v][u] = weight

        return Matrix(min_span_tree)

    def dijkstra(self, n, m):
        if self.rows != self.cols:
            raise ValueError("Для алгоритма Дейкстры матрица должна быть квадратной.")
        if n < 0 or n >= self.rows or m < 0 or m >= self.rows:
            raise ValueError("Неверные индексы вершин")

        # Инициализация
        distances = [float('inf')] * self.rows
        distances[n] = 0
        visited = [False] * self.rows
        previous = [-1] * self.rows

        for _ in range(self.rows):
            # Находим вершину с минимальным расстоянием
            min_dist = float('inf')
            u = -1
            for i in range(self.rows):
                if not visited[i] and distances[i] < min_dist:
                    min_dist = distances[i]
                    u = i

            if u == -1:
                break

            visited[u] = True

            # Обновляем расстояния до соседей
            for v in range(self.rows):
                if self.matrix[u][v] > 0 and not visited[v]:
                    alt = distances[u] + self.matrix[u][v]
                    if alt < distances[v]:
                        distances[v] = alt
                        previous[v] = u

        # Восстанавливаем путь
        path = []
        current = m
        while current != -1:
            path.append(current)
            current = previous[current]
        path.reverse()

        if distances[m] == float('inf'):
            return None, float('inf')  # Путь не существует
        else:
            return path, distances[m]

    def gaussian(self):
        if self.rows != self.cols - 1:
            raise ValueError("Для решения линейных уравнений матрица должна иметь n строк и n+1 столбцов.")

        n = self.rows
        mat = [row[:] for row in self.matrix]  # Создаем копию матрицы

        # Прямой ход метода Гаусса
        for col in range(n):
            # Поиск строки с максимальным элементом в текущем столбце
            max_row = col
            for i in range(col + 1, n):
                if abs(mat[i][col]) > abs(mat[max_row][col]):
                    max_row = i

            # Обмен строк
            mat[col], mat[max_row] = mat[max_row], mat[col]

            # Проверка на ноль на диагонали
            if mat[col][col] == 0:
                raise ValueError("Система не имеет единственного решения.")

            # Нормализация текущей строки
            div = mat[col][col]
            for j in range(col, n + 1):
                mat[col][j] /= div

            # Исключение переменной из других строк
            for i in range(n):
                if i != col and mat[i][col] != 0:
                    factor = mat[i][col]
                    for j in range(col, n + 1):
                        mat[i][j] -= factor * mat[col][j]

        # Обратный ход (извлечение решения)
        solution = [0] * n
        for i in range(n):
            solution[i] = mat[i][n]

        return solution