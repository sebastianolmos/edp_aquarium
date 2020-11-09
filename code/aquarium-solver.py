import json
import numpy as np
import matplotlib.pyplot as mpl
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import sys

#Se recibe como parametro la configuracion en un archivo json
file_name = sys.argv[1]
file = open(file_name)
data = json.load(file)

if __name__ == "__main__":

    H = data["height"]
    W = data["width"]
    D = data["lenght"]
    W_LOSS = data["window_loss"]
    H_A = data["heater_a"]
    H_B = data["heater_b"]
    AMBIENT_T = data["ambient_temperature"]


    #h ajustado para las dimensiones
    max_side = max(W, D, H)
    h=max_side/30

    # Number of unknowns
    nh = int(W / h) + 1 # x
    nd = int(D / h) + 1 # y
    nv = int(H / h) - 0 # z

    # In this case, the domain is just a rectangular cube
    N = nh * nv * nd

    # We define a function to convert the indices from i,j, k to q and viceversa
    # i,j, k indexes the discrete domain in 3D.
    # q parametrize those i,j,k, this way we can tidy the unknowns
    # in a column vector and use the standard algebra

    def getQ(i, j, k):
        return i + nh * (j + nd * k)

    def getIJK(q):
        i = (q % (nd * nh)) % nh
        j = (q % (nd * nh)) // nh
        k = q // (nd * nh)
        return (i, j, k)

    # In this matrix we will write all the coefficients of the unknowns
    A = lil_matrix((N, N))

    # In this vector we will write all the right side of the equations
    b = np.zeros((N,))

    # Note: To write an equation is equivalent to write a row in the matrix system

    # We iterate over each point inside the domain
    # Each point has an equation associated
    # The equation is different depending on the point location inside the domain
    for i in range(0, nh):
        for j in range(0, nd):
            for k in range(0, nv):
                # We will write the equation associated with row q
                q = getQ(i, j, k)

                # We obtain indices of the other coefficients
                q_up = getQ(i, j, k + 1)
                q_down = getQ(i, j, k - 1)
                q_left = getQ(i - 1, j, k)
                q_right = getQ(i + 1, j, k)
                q_forward = getQ(i, j + 1, k)
                q_back = getQ(i, j - 1, k)

                # Depending on the location of the point, the equation is different

                # Interior # 0
                if 1 <= i and i <= nh - 2 and 1 <= j and j <= nd - 2 and 1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = 0

                # Upper Face # 1, WATER SURFACE
                elif 1 <= i and i <= nh - 2 and 1 <= j and j <= nd - 2 and k == nv - 1:
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -AMBIENT_T

                # Left Face # 2, WINDOW
                elif 1 <= k and k <= nv - 2 and 1 <= j and j <= nd - 2 and i == 0:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS

                # Lower Face # 3, FLOOR NEUMANN
                elif 1 <= i and i <= nh - 2 and 1 <= j and j <= nd - 2 and k == 0:
                    #HEATER A
                    if i >= 1*nh//3 and i < 2*nh//3 and j >= 3*nd//5 and j < 4*nd//5:
                        A[q, q] = 1
                        b[q] = H_A
                    #HEATER B
                    elif i >= 1*nh//3 and i < 2*nh//3 and j >= 1*nd//5 and j < 2*nd//5:
                        A[q, q] = 1
                        b[q] = H_B
                    else:
                        A[q, q_up] = 2
                        A[q, q_left] = 1
                        A[q, q_right] = 1
                        A[q, q_forward] = 1
                        A[q, q_back] = 1
                        A[q, q] = -6
                        b[q] = -2 * h * 0

                # Right Face # 4, WINDOW
                elif 1 <= k and k <= nv - 2 and 1 <= j and j <= nd - 2 and i == nh - 1:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS

                # Back Face # 5, WINDOW
                elif 1 <= i and i <= nh - 2 and j == 0 and 1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS

                # Forward Face #6, WINDOW
                elif 1 <= i and i <= nh - 2 and j == nd - 1 and 1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS

                # Left Lower Edge # 7
                elif (i, k) == (0, 0) and  1 <= j and j <= nd - 2:
                    A[q, q_up] = 2
                    A[q, q_right] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS  -2 * h* 0

                #Right Lower Edge # 8
                elif (i, k) == (nh - 1, 0) and  1 <= j and j <= nd - 2:
                    A[q, q_up] = 2
                    A[q, q_left] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS  -2 * h* 0

                #Right Upper Edge # 9
                elif (i, k) == (nh - 1, nv - 1) and  1 <= j and j <= nd - 2:
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -AMBIENT_T

                # Left Upper Edge # 10
                elif (i, k) == (0, nv - 1) and  1 <= j and j <= nd - 2:
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_forward] = 1
                    A[q, q_back] = 1
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -AMBIENT_T

                #Left Back Edge # 11
                elif (i, j) == (0, 0) and  1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -2 * h* W_LOSS

                # Lower Back Edge # 12
                elif (j, k) == (0, 0) and  1 <= i and i <= nh - 2:
                    A[q, q_up] = 2
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS  -2 * h* 0

                # Right Back Edge # 13
                elif (i, j) == (nh - 1, 0) and  1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -2 * h* W_LOSS

                # Upper Back Edge # 14
                elif (j, k) == (0, nv - 1) and  1 <= i and i <= nh - 2:
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS  -AMBIENT_T

                #Left Forward Edge # 15
                elif (i, j) == (0, nd - 1) and  1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -2 * h* W_LOSS

                # Lower Forward Edge # 16
                elif (j, k) == (nd - 1, 0) and  1 <= i and i <= nh - 2:
                    A[q, q_up] = 2
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -2 * h* 0

                # Right Forward Edge # 17
                elif (i, j) == (nh - 1, nd - 1) and  1 <= k and k <= nv - 2:
                    A[q, q_up] = 1
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS -2 * h* W_LOSS

                # Upper Forward Edge # 18
                elif (j, k) == (nd - 1, nv - 1) and  1 <= i and i <= nh - 2:
                    A[q, q_down] = 1
                    A[q, q_left] = 1
                    A[q, q_right] = 1
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* W_LOSS  -AMBIENT_T

                # Left Lower Back Corner # 19
                elif (i, j, k) == (0, 0, 0):
                    A[q, q_up] = 2
                    A[q, q_right] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* 0 -2 * h* W_LOSS -2 * h* W_LOSS

                # Right Lower Back Corner # 20
                elif (i, j, k) == (nh - 1, 0, 0):
                    A[q, q_up] = 2
                    A[q, q_left] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* 0 -2 * h* W_LOSS -2 * h* W_LOSS

                # Right Upper Back Corner # 21
                elif (i, j, k) == (nh - 1, 0, nv - 1):
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -AMBIENT_T -2 * h* W_LOSS -2 * h* W_LOSS

                # Left Upper Back Corner # 22
                elif (i, j, k) == (0, 0, nv - 1):
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_forward] = 2
                    A[q, q] = -6
                    b[q] = -AMBIENT_T -2 * h* W_LOSS -2 * h* W_LOSS

                # Left Lower Forward Corner # 23
                elif (i, j, k) == (0, nd - 1, 0):
                    A[q, q_up] = 2
                    A[q, q_right] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* 0 -2 * h* W_LOSS -2 * h* W_LOSS

                # Right Lower Forward Corner # 24
                elif (i, j, k) == (nh - 1, nd - 1, 0):
                    A[q, q_up] = 2
                    A[q, q_left] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -2 * h* 0 -2 * h* W_LOSS -2 * h* W_LOSS

                # Right Upper Forward Corner # 25
                elif (i, j, k) == (nh - 1, nd - 1, nv - 1):
                    A[q, q_down] = 1
                    A[q, q_left] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -AMBIENT_T -2 * h* W_LOSS -2 * h* W_LOSS

                # Left Upper Forward Corner # 26
                elif (i, j, k) == (0, nd - 1, nv - 1):
                    A[q, q_down] = 1
                    A[q, q_right] = 2
                    A[q, q_back] = 2
                    A[q, q] = -6
                    b[q] = -AMBIENT_T -2 * h* W_LOSS -2 * h* W_LOSS

                else:
                    print("Point (" + str(i) + ", " + str(j) + ", " + str(k) +") missed!")
                    print("Associated point index is " + str(q))
                    raise Exception()

    # Solving our matrix with a Sparse matrix
    x = spsolve(A, b)

    # Now we return our solution to the 3d discrete domain
    # In this matrix we will store the solution in the 3d domain
    u = np.zeros((nh, nd, nv))

    for q in range(0, N):
        i, j, k = getIJK(q)
        u[i, j, k] = x[q]

    # Adding the borders(faces), as they have known values
    ub = np.zeros((nh, nd, nv + 1))
    ub[0:nh + 0, 0:nd + 0,  0:nv ] = u[:, :, :]

    # Dirichlet boundary condition
    # Ceil Face
    ub[0:nh+0, 0:nd+0, nv] = AMBIENT_T

    np.save(data["filename"], ub)

    #VISTA 3D CON SCATTER
    '''
    X, Y, Z = np.mgrid[0:W:np.complex(0, ub.shape[0]), 0:D:np.complex(0, ub.shape[1]), 0:H:np.complex(0, ub.shape[2])]

    fig = mpl.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')

    #Show the values in a 3d discrete domain
    scat = ax.scatter(X, Y, Z, c=ub, alpha=0.2, s=80, marker='s',cmap='Spectral' )

    fig.colorbar(scat, shrink=0.5, aspect=5)  # This is the colorbar at the side

    # Showing the result
    ax.set_title('Aquarium solution scatter view')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    mpl.show()
    '''
    #VISTA 2D DESDE UN EJE
    X, Y, Z = np.mgrid[0:W:np.complex(0, ub.shape[0]), 0:D:np.complex(0, ub.shape[1]), 0:H:np.complex(0, ub.shape[2])]
    print(X.shape)
    print(Y.shape)
    print(Z.shape)

    fig, ax = mpl.subplots(1, 1)
    pcm = ax.pcolormesh(ub[X.shape[0]//2].T, cmap='coolwarm')
    fig.colorbar(pcm)
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title('Aquarium solution 2d center view ')
    ax.set_aspect('equal', 'datalim')

    # Note:
    # imshow is also valid but it uses another coordinate system,
    # a data transformation is required
    # ax.imshow(ub.T)
    mpl.show()