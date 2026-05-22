from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import DTypeLike
from scipy.io import mmread

from practicum_9.lu import LinearSystemSolver
from src.common import NDArrayFloat


class LuSolverWithPermute(LinearSystemSolver):
    def __init__(self, A: NDArrayFloat, dtype: DTypeLike, permute: bool) -> None:
        super().__init__(A, dtype)
        self.L, self.U, self.P = self._decompose(permute)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        n = len(b)
        b_perm = self.P @ b
        y = np.zeros(n, dtype=self.dtype)
        for i in range(n):
            y[i] = b_perm[i] - np.dot(self.L[i, :i], y[:i])
        x = np.zeros(n, dtype=self.dtype)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(self.U[i, i+1:], x[i+1:])) / self.U[i, i]
        return x

    def _decompose(self, permute: bool) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        n = self.A.shape[0]
        A = self.A.copy().astype(self.dtype)
        L = np.eye(n, dtype=self.dtype)
        U = np.zeros((n, n), dtype=self.dtype)
        P = np.eye(n, dtype=self.dtype)

        if permute:
            for i in range(n-1):
                pivot_row = i + np.argmax(np.abs(A[i:, i]))
                if pivot_row != i:
                    A[[i, pivot_row]] = A[[pivot_row, i]]
                    L[[i, pivot_row]] = L[[pivot_row, i]]
                    P[[i, pivot_row]] = P[[pivot_row, i]]

                for k in range(i+1, n):
                    factor = A[k, i] / A[i, i]
                    L[k, i] = factor
                    A[k, i:] -= factor * A[i, i:]
        else:
            for i in range(n-1):
                if np.abs(A[i, i]) == 0:
                    raise ValueError("Zero pivot")
                for k in range(i+1, n):
                    factor = A[k, i] / A[i, i]
                    L[k, i] = factor
                    A[k, i:] -= factor * A[i, i:]

        U = np.triu(A).astype(self.dtype)
        return L, U, P


def load_matrix_and_rhs(filename):
    """Загружает матрицу из .mtx файла и создаёт правую часть b = A * (1,1,...,1)^T."""
    A = mmread(filename).tocsc()        # читаем разрежённую матрицу
    A_dense = A.toarray()               # преобразуем в плотную (ваш решатель требует плотную)
    n = A.shape[0]
    x_exact = np.ones(n, dtype=np.float64)
    b = A_dense @ x_exact
    return A_dense, b, x_exact


def get_A_b(a_11: float, b_1: float) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Исходная тестовая матрица 3x3 (для проверки)."""
    A = np.array([[a_11, 1.0, -3.0], [6.0, 2.0, 5.0], [1.0, 4.0, -3.0]])
    b = np.array([b_1, 12.0, -39.0])
    return A, b


if __name__ == "__main__":
    # Сначала тест на маленькой матрице (как раньше)
    print("=== Тест на матрице 3x3 ===")
    p = 16
    a_11 = 3 + 10 ** (-p)
    b_1 = -16 + 10 ** (-p)
    A, b = get_A_b(a_11, b_1)
    solver = LuSolverWithPermute(A, np.float64, permute=True)
    x = solver.solve(b)
    assert np.all(np.isclose(x, [1, -7, 4])), f"The answer {x} is not accurate enough"
    print("Тест 3x3 пройден успешно.")

    files = [
        "data/mcca.mtx",
        "data/mcfe.mtx",
        "data/bcsstk14.mtx",
    ]

    print("\n=== Тест на матрицах из файлов ===")
    for fname in files:
        print(f"\n--- Решение для {fname} ---")
        try:
            A, b, x_exact = load_matrix_and_rhs(fname)
            print(f"Размер матрицы: {A.shape}")
            solver = LuSolverWithPermute(A, np.float64, permute=True)
            x_computed = solver.solve(b)
            rel_err = np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)
            resid = np.linalg.norm(A @ x_computed - b)
            print(f"Относительная ошибка: {rel_err:.2e}")
            print(f"Невязка: {resid:.2e}")
        except FileNotFoundError:
            print(f"Файл {fname} не найден. Убедитесь, что он лежит в папке data.")
        except Exception as e:
            print(f"Ошибка при решении: {e}")