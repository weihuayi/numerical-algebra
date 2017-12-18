import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def foo(axes, A=None, n=None):
    """
    :param axes:
    :param A: any symmetric 3-by-3 real matrix
    :param n: level number of Rayleigh quotient

    denote [k1, k2, k3] to be the eigenvalue of matrix A, k1 < k2 < k3

    set seq = [i1, ..., in] + [k2] + [j1, ..., jn]
    set cseq = ['g'] * n + ['k'] + ['r'] * n
    then draw the lines with ||u|| == 1, u.T @ A @ u == seq[i] and color == sceq[i]
    """
    phi, theta = np.mgrid[-np.pi/2:np.pi/2:60j, 0:2*np.pi:120j]

    xx = np.cos(phi) * np.cos(theta)
    yy = np.cos(phi) * np.sin(theta)
    zz = np.sin(phi)

    axes.plot_surface(xx, yy, zz, color='b', alpha=0.2)

    if A is None:
        B = np.random.rand(3, 3)
        A = B.T + B

    u = np.vstack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)))
    aa = np.diag((np.dot(u.T, A)).dot(u)).reshape(xx.shape)

    eigvals = sorted(list(np.linalg.eigvals(A)))
    if n is None or n == 0:
        seq = eigvals[1:-1]
        cseq = ['k']
    else:
        if len(np.linalg.eigvals(A)) == 1:
            seq = eigvals[1:-1]
            cseq = ['k']
        else:
            seq = list(np.linspace(eigvals[0], eigvals[1], n + 1, endpoint=False))[1:]
            cseq = ['g'] * n
            seq += list(np.linspace(eigvals[1], eigvals[2], n + 1, endpoint=False))
            cseq += ['k'] + ['r'] * n

    num = len(axes.collections)
    cs = axes.contour3D(phi, theta, aa, seq)
    segs = cs.allsegs
    for _ in range(len(axes.collections) - num):
        axes.collections.pop(-1)

    for i in range(len(segs)):
        c = cseq[i]
        for line in segs[i]:
            x = np.cos(line[:, 0]) * np.cos(line[:, 1])
            y = np.cos(line[:, 0]) * np.sin(line[:, 1])
            z = np.sin(line[:, 0])
            axes.plot3D(x, y, z, c=c)


if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 6))

    ax = axes3d.Axes3D(fig)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    foo(axes=ax, A=None, n=5)

    plt.show()

