import unittest
from matrix_operations import Matrix
import sys


class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        self.matrix1 = Matrix([[1, 2], [3, 4]])
        self.matrix2 = Matrix([[5, 6], [7, 8]])
        self.matrix3 = Matrix([[1, 2, 3], [4, 5, 6]])
        self.matrix4 = Matrix([[1, 0], [0, 1]])
        self.matrix5 = Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.graph_matrix = Matrix([
            [0, 2, 0, 6, 0],
            [2, 0, 3, 8, 5],
            [0, 3, 0, 0, 7],
            [6, 8, 0, 0, 9],
            [0, 5, 7, 9, 0]
        ])
        self.equation_matrix = Matrix([
            [2, 1, -1, 8],
            [-3, -1, 2, -11],
            [-2, 1, 2, -3]
        ])

    def test_addition(self):
        """Тестирование сложения матриц"""
        print("\nЗапуск test_addition...")
        result = self.matrix1 + self.matrix2
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(result.matrix, expected.matrix)
        print("Сложение матриц одинакового размера: УСПЕХ")

        with self.assertRaises(ValueError):
            self.matrix1 + self.matrix3
        print("Проверка ошибки при сложении разных размеров: УСПЕХ")

    def test_subtraction(self):
        """Тестирование вычитания матриц"""
        print("\nЗапуск test_subtraction...")
        result = self.matrix2 - self.matrix1
        expected = Matrix([[4, 4], [4, 4]])
        self.assertEqual(result.matrix, expected.matrix)
        print("Вычитание матриц одинакового размера: УСПЕХ")

        with self.assertRaises(ValueError):
            self.matrix1 - self.matrix3
        print("Проверка ошибки при вычитании разных размеров: УСПЕХ")

    def test_scalar_multiplication(self):
        """Тестирование умножения матрицы на скаляр"""
        print("\nЗапуск test_scalar_multiplication...")
        result = self.matrix1 * 3
        expected = Matrix([[3, 6], [9, 12]])
        self.assertEqual(result.matrix, expected.matrix)
        print("Умножение матрицы на скаляр: УСПЕХ")

    def test_matrix_multiplication(self):
        """Тестирование умножения матриц"""
        print("\nЗапуск test_matrix_multiplication...")
        result = self.matrix1 * self.matrix2
        expected = Matrix([[19, 22], [43, 50]])
        self.assertEqual(result.matrix, expected.matrix)
        print("Умножение матриц 2x2: УСПЕХ")

    def test_determinant(self):
        """Тестирование вычисления определителя"""
        print("\nЗапуск test_determinant...")
        self.assertEqual(self.matrix1.determinant(), -2)
        print("Определитель матрицы 2x2: УСПЕХ")

        self.assertEqual(self.matrix4.determinant(), 1)
        print("Определитель единичной матрицы: УСПЕХ")

        with self.assertRaises(ValueError):
            self.matrix3.determinant()
        print("Проверка ошибки для неквадратной матрицы: УСПЕХ")

    def test_is_connected(self):
        """Тестирование проверки связности графа"""
        print("\nЗапуск test_is_connected...")
        self.assertTrue(self.matrix5.is_connected())
        print("Связный граф 3x3: УСПЕХ")

        self.assertTrue(Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).is_connected())
        print("Другой связный граф: УСПЕХ")

        self.assertFalse(Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).is_connected())
        print("Несвязный граф: УСПЕХ")

        with self.assertRaises(ValueError):
            self.matrix3.is_connected()
        print("Проверка ошибки для неквадратной матрицы: УСПЕХ")

    def test_kruskal(self):
        """Тестирование алгоритма Краскала"""
        print("\nЗапуск test_kruskal...")
        min_span_tree = self.graph_matrix.kruskal()
        expected_edges = [
            (0, 1, 2),
            (1, 2, 3),
            (1, 4, 5),
            (0, 3, 6)
        ]
        total_weight = sum(edge[2] for edge in expected_edges)

        actual_weight = sum(min_span_tree.matrix[i][j] for i in range(min_span_tree.rows)
                            for j in range(i + 1, min_span_tree.cols))
        self.assertEqual(actual_weight, total_weight)
        print("Алгоритм Краскала: УСПЕХ (правильный суммарный вес)")

    def test_dijkstra(self):
        """Тестирование алгоритма Дейкстры"""
        print("\nЗапуск test_dijkstra...")
        path, distance = self.graph_matrix.dijkstra(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(distance, 5)
        print("Кратчайший путь 0->2: УСПЕХ")

        path, distance = self.graph_matrix.dijkstra(0, 4)
        self.assertEqual(path, [0, 1, 4])
        self.assertEqual(distance, 7)
        print("Кратчайший путь 0->4: УСПЕХ")

        disconnected_graph = Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertIsNone(disconnected_graph.dijkstra(0, 3)[0])
        print("Несвязный граф: УСПЕХ (путь не существует)")

    def test_gaussian(self):
        """Тестирование метода Гаусса"""
        print("\nЗапуск test_gaussian...")
        solution = self.equation_matrix.gaussian()
        expected = [2, 3, -1]
        self.assertAlmostEqual(solution[0], expected[0], places=5)
        self.assertAlmostEqual(solution[1], expected[1], places=5)
        self.assertAlmostEqual(solution[2], expected[2], places=5)
        print("Решение системы уравнений: УСПЕХ")

        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4]]).gaussian()
        print("Проверка ошибки для некорректной системы: УСПЕХ")


def main():
    print("=== НАЧАЛО ТЕСТИРОВАНИЯ ===")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMatrixOperations)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n=== ИТОГОВЫЙ ОТЧЕТ ===")
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")

    if result.failures:
        print("\nПроваленные тесты:")
        for test, traceback in result.failures:
            print(f"- {test.id()}")

    if result.errors:
        print("\nТесты с ошибками:")
        for test, traceback in result.errors:
            print(f"- {test.id()}")

    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")
    print("Результат: " + ("ОШИБКИ" if not result.wasSuccessful() else "ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО"))


if __name__ == '__main__':
    main()