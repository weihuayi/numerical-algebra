import numpy as np
import copy


def foo(A):
    B = np.eye(n, dtype=float)

    for i in range(n-1):
        for j in range(i+1):
            B[i, j] /= A[i, i]

        for j in range(i+1, n):
            A[i, j] /= A[i, i]
        A[i, i] = 1

        for k in range(i+1, n):
            for j in range(i):
                B[k, j] -= B[i, j] * A[k, i]

            B[k, i] = -B[i, i] * A[k, i]

            for j in range(i+1, n):
                A[k, j] -= A[i, j] * A[k, i]

            A[k, i] = 0

    for j in range(n):
        B[n-1, j] /= A[n-1, n-1]
    A[n-1, n-1] = 1

    for i in range(n-2, -1, -1):
        if i == n-2:
            for k in range(i, -1, -1):
                for j in range(k+1):
                    B[k, j] -= B[i+1, j]*A[k, i+1]
                for j in range(k+1, n):
                    B[k, j] = -B[i+1, j]*A[k, i+1]

                A[k, i+1] = 0
        else:
            for k in range(i, -1, -1):
                for j in range(n):
                    B[k, j] -= B[i+1, j]*A[k, i+1]

                A[k, i+1] = 0

    return B


n = 6
A = np.random.randint(low=0, high=n*n, size=(n, n)) + 0.
B = foo(copy.deepcopy(A))
ans = B.dot(A)

for i in range(n):
    for j in range(n):
        if abs(ans[i, j]) < 1e-6:
            ans[i, j] = 0
        elif abs(ans[i, j] - 1) < 1e-6:
            ans[i, j] = 1
print('A = \n', A)
print('B = \n', B)
print('B.dot(A) = \n', ans)
