from OpenGL.GL import *
import numpy as np
import sys
from PIL import Image

import transformations as tr
import easy_shaders as es
import basic_shapes as bs

#Funcion para cargar textura con MipMap
def textureWithMipMapSetup(texture, imgName):
    glBindTexture(GL_TEXTURE_2D, texture)

    # texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    image = Image.open(imgName)
    img_data = np.array(list(image.getdata()), np.uint8)

    if image.mode == "RGB":
        internalFormat = GL_RGB
        format = GL_RGB
    elif image.mode == "RGBA":
        internalFormat = GL_RGBA
        format = GL_RGBA
    else:
        print("Image mode not supported.")
        raise Exception()

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image.size[0], image.size[1], 0, format, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

def NormalCube(r = 1):
    r /= 0.5
    vertices = [
        #   positions         colors   normals
        # Z-
        r, r, -r, 0, 0, -1,
        r, -r, -r, 0, 0, -1,
        -r, -r, -r,  0, 0, -1,
        -r, r, -r, 0, 0, -1,

        # Z+
        -r, -r, r, 0, 0, 1,
        r, -r, r, 0, 0, 1,
        r, r, r, 0, 0, 1,
        -r, r, r, 0, 0, 1,

        # X-
        -r, r, -r, -1, 0, 0,
        -r, -r, -r, -1, 0, 0,
        -r, -r, r, -1, 0, 0,
        -r, r, r, -1, 0, 0,

        # X+
        r, -r, -r, 1, 0, 0,
        r, r, -r, 1, 0, 0,
        r, r, r, 1, 0, 0,
        r, -r, r, 1, 0, 0,

        # Y-
        -r, -r, -r, 0, -1, 0,
        r, -r, -r, 0, -1, 0,
        r, -r, r, 0, -1, 0,
        -r, -r, r, 0, -1, 0,

        # Y+
        r, r, -r, 0, 1, 0,
        -r, r, -r, 0, 1, 0,
        -r, r, r, 0, 1, 0,
        r, r, r, 0, 1, 0,

    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,  # Z+
        4, 5, 6, 6, 7, 4,  # Z-
        8, 9, 10, 10, 11, 8,  # X+
        12, 13, 14, 14, 15, 12,  # X-
        16, 17, 18, 18, 19, 16,  # Y+
        20, 21, 22, 22, 23, 20]  # Y-

    return bs.Shape(vertices, indices, None)

def NormalCub3e(r = 1):
    r /= 0.5
    vertices = [
        #   positions         colors   normals
        # Z+
        -r, -r, r, 0, 0, 1,
        r, -r, r, 0, 0, 1,
        r, r, r, 0, 0, 1,
        -r, r, r, 0, 0, 1,

        # Z-
        -r, -r, -r, 0, 0, -1,
        r, -r, -r, 0, 0, -1,
        r, r, -r,  0, 0, -1,
        -r, r, -r, 0, 0, -1,

        # X+
        r, -r, -r, 1, 0, 0,
        r, r, -r, 1, 0, 0,
        r, r, r, 1, 0, 0,
        r, -r, r, 1, 0, 0,

        # X-
        -r, -r, -r,  -1, 0, 0,
        -r, r, -r, -1, 0, 0,
        -r, r, r, -1, 0, 0,
        -r, -r, r, -1, 0, 0,

        # Y+
        -r, r, -r, 0, 1, 0,
        r, r, -r, 0, 1, 0,
        r, r, r, 0, 1, 0,
        -r, r, r, 0, 1, 0,

        # Y-
        -r, -r, -r, 0, -1, 0,
        r, -r, -r, 0, -1, 0,
        r, -r, r, 0, -1, 0,
        -r, -r, r, 0, -1, 0
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,  # Z+
        7, 6, 5, 5, 4, 7,  # Z-
        8, 9, 10, 10, 11, 8,  # X+
        15, 14, 13, 13, 12, 15,  # X-
        19, 18, 17, 17, 16, 19,  # Y+
        20, 21, 22, 22, 23, 20]  # Y-

    return bs.Shape(vertices, indices, None)
#Funcion para crear la data de los voxeles con una determinada temperatura
def create_solid_zone(sol, t_i, c):
    d = 0.5 #Color offset
    #Colores para los vertices
    c1 = [min(c[0] + d, 1.0), min(c[1] + d, 1.0), min(c[2] + d, 1.0)]
    c2 = [max(c[0] - d, 0.0), max(c[1] - d, 0.0), max(c[2] - d, 0.0)]

    vertices = []
    indices = []
    #Dimensiones
    x_len = sol.shape[0] - 1
    y_len = sol.shape[1] - 1
    z_len = sol.shape[2] - 1
    #Matriz del acuario para indicar con 1s los indices en que los voxeles se encuentren en el tango de temperatura
    in_voxel = np.zeros(sol.shape)
    #Arreglo para guardar la posiciones de los voxeles que se encuentren en el rango
    voxel_pos = []
    v_counter = 0
    #recorremos las matriz solucion buscando los indices que se encuentren en el rango, luego desde estos voxeles,
    #se visitan los vecinos para saber si estos se encuentran en el rango, si no se encuentra en el rango, se creara
    #un cuadrado entre ambos voxeles, de esta manera solo se crean las caras en los limites de la zona de Temperatura
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            for k in range(sol.shape[2]):
                temp_value = sol[i, j, k]
                if temp_value >= t_i - 2 and temp_value <= t_i + 2:
                    #Se encontro un voxel dentro del rango
                    voxel_pos += [np.array([i+0.5, j+0.5, k+0.5])] #se agrega la posicion del voxel
                    in_voxel[i,j,k] = 1 # se rellena con 1

                    #Comprobamos si los vecinos se encuentran en el rango pra dibujar solo las caras externas
                    # EJE X
                    if i != 0:
                        near_value = sol[i - 1, j, k]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0, j + 1, k + 0, c[0], c[1], c[2],
                                i + 0, j + 0, k + 0, c2[0], c2[1], c2[2],
                                i + 0, j + 0, k + 1, c[0], c[1], c[2],
                                i + 0, j + 1, k + 1, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0, j + 1, k + 0, c[0], c[1], c[2],
                            i + 0, j + 0, k + 0, c2[0], c2[1], c2[2],
                            i + 0, j + 0, k + 1, c[0], c[1], c[2],
                            i + 0, j + 1, k + 1, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if i != x_len:
                        near_value = sol[i + 1, j, k]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 1, j + 0, k + 0, c[0], c[1], c[2],
                                i + 1, j + 1, k + 0, c2[0], c2[1], c2[2],
                                i + 1, j + 1, k + 1, c[0], c[1], c[2],
                                i + 1, j + 0, k + 1, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 1, j + 0, k + 0, c[0], c[1], c[2],
                            i + 1, j + 1, k + 0, c2[0], c2[1], c2[2],
                            i + 1, j + 1, k + 1, c[0], c[1], c[2],
                            i + 1, j + 0, k + 1, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    # EJE Y
                    if j != 0:
                        near_value = sol[i, j - 1, k]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0, j + 0, k + 0, c[0], c[1], c[2],
                                i + 1, j + 0, k + 0, c2[0], c2[1], c2[2],
                                i + 1, j + 0, k + 1, c[0], c[1], c[2],
                                i + 0, j + 0, k + 1, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0, j + 0, k + 0, c[0], c[1], c[2],
                            i + 1, j + 0, k + 0, c2[0], c2[1], c2[2],
                            i + 1, j + 0, k + 1, c[0], c[1], c[2],
                            i + 0, j + 0, k + 1, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if j != y_len:
                        near_value = sol[i, j + 1, k]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 1, j + 1, k + 0, c[0], c[1], c[2],
                                i + 0, j + 1, k + 0, c2[0], c2[1], c2[2],
                                i + 0, j + 1, k + 1, c[0], c[1], c[2],
                                i + 1, j + 1, k + 1, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 1, j + 1, k + 0, c[0], c[1], c[2],
                            i + 0, j + 1, k + 0, c2[0], c2[1], c2[2],
                            i + 0, j + 1, k + 1, c[0], c[1], c[2],
                            i + 1, j + 1, k + 1, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    # EJE Z
                    if k != 0:
                        near_value = sol[i, j, k-1]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0, j + 1, k + 0, c[0], c[1], c[2],
                                i + 1, j + 1, k + 0, c2[0], c2[1], c2[2],
                                i + 1, j + 0, k + 0, c[0], c[1], c[2],
                                i + 0, j + 0, k + 0, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0, j + 1, k + 0, c[0], c[1], c[2],
                            i + 1, j + 1, k + 0, c2[0], c2[1], c2[2],
                            i + 1, j + 0, k + 0, c[0], c[1], c[2],
                            i + 0, j + 0, k + 0, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if k != z_len:
                        near_value = sol[i, j, k+1]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0, j + 0, k + 1, c[0], c[1], c[2],
                                i + 1, j + 0, k + 1, c2[0], c2[1], c2[2],
                                i + 1, j + 1, k + 1, c[0], c[1], c[2],
                                i + 0, j + 1, k + 1, c1[0], c1[1], c1[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0, j + 0, k + 1, c[0], c[1], c[2],
                            i + 1, j + 0, k + 1, c2[0], c2[1], c2[2],
                            i + 1, j + 1, k + 1, c[0], c[1], c[2],
                            i + 0, j + 1, k + 1, c1[0], c1[1], c1[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4
    if voxel_pos == []:
        #Si no se encuentran voxeles en el rango de temperatura dado, no se puede crear la zona
        print("VOXELS WITHOUT T°"+ str(t_i) + " RANGE")
    return bs.Shape(vertices, indices), voxel_pos, in_voxel

#Funcion similar a la anterior pero que solo crea la shape de 1 color y con un offset para poder escalarse y dibujar la outline
def create_outline_zone(sol, t_i, c, s = 0.2):
    vertices = []
    indices = []
    x_len = sol.shape[0] - 1
    y_len = sol.shape[1] - 1
    z_len = sol.shape[2] - 1

    v_counter = 0
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            for k in range(sol.shape[2]):
                temp_value = sol[i, j, k]
                if temp_value >= t_i - 2 and temp_value <= t_i + 2:
                    # EJE X
                    if i != 0:
                        near_value = sol[i - 1, j, k]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 0 - s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                                i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                                i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 0 - s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                            i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                            i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if i != x_len:
                        near_value = sol[i + 1, j, k]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 1 + s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                                i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 1 + s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                            i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    # EJE Y
                    if j != 0:
                        near_value = sol[i, j - 1, k]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0 - s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                                i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0 - s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j + 0 - s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                            i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if j != y_len:
                        near_value = sol[i, j + 1, k]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                                i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                            i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    # EJE Z
                    if k != 0:
                        near_value = sol[i, j, k-1]
                        if not(near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                                i + 1 + s, j - 0 - s, k - 0 - s, c[0], c[1], c[2],
                                i + 0 - s, j - 0 - s, k - 0 - s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0 - s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j + 1 + s, k + 0 - s, c[0], c[1], c[2],
                            i + 1 + s, j - 0 - s, k - 0 - s, c[0], c[1], c[2],
                            i + 0 - s, j - 0 - s, k - 0 - s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

                    if k != z_len:
                        near_value = sol[i, j, k+1]
                        if not (near_value >= t_i - 2 and near_value <= t_i + 2):
                            vertices += [
                                i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                                i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                                i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                                i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                            ]
                            indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                        v_counter + 2, v_counter + 3, v_counter + 0]
                            v_counter += 4
                    else:
                        vertices += [
                            i + 0 - s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                            i + 1 + s, j + 0 - s, k + 1 + s, c[0], c[1], c[2],
                            i + 1 + s, j + 1 + s, k + 1 + s, c[0], c[1], c[2],
                            i + 0 - s, j + 1 + s, k + 1 + s, c[0], c[1], c[2]
                        ]
                        indices += [v_counter + 0, v_counter + 1, v_counter + 2,
                                    v_counter + 2, v_counter + 3, v_counter + 0]
                        v_counter += 4

    return bs.Shape(vertices, indices)

#Funcion para crear los elementos de la sala
def create_room(dim, rad, sub_h, up_h):
    W, D, H = dim[0], dim[1], dim[2]
    shapes = []

    # Mueble donde se soporta el acuario
    # deltas para los texeles
    dx = (W / min(W, D, H))/2
    dy = (D/ min(W, D, H))/2
    dz = (H / min(W, D, H))/2
    vertices_0 = [
        0, 0, -sub_h, 0, dz, 0, -1, 0, # 0
        W, 0, -sub_h, dx, dz, 0, -1, 0, # 1
        W, 0, 0, dx, 0, 0, -1, 0,      # 2
        0, 0, 0, 0, 0, 0, -1, 0,      # 3

        W, 0, -sub_h, 0, dz, 1, 0, 0,  # 4
        W, D, -sub_h, dy, dz, 1, 0, 0,  # 5
        W, D, 0, dy, 0, 1, 0, 0,       # 6
        W, 0, 0, 0, 0, 1, 0, 0,       # 7

        W, D, -sub_h, 0, dz, 0, 1, 0,  # 8
        0, D, -sub_h, dx, dz, 0, 1, 0,  # 9
        0, D, 0, dx, 0, 0, 1, 0,       # 10
        W, D, 0, 0, 0, 0, 1, 0,       # 11

        0, D, -sub_h, 0, dz, -1, 0, 0, # 12
        0, 0, -sub_h, dy, dz, -1, 0, 0, # 13
        0, 0, 0, dy, 0, -1, 0, 0,      # 14
        0, D, 0, 0, 0, -1, 0, 0       # 15
    ]
    indices_0 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12
    ]
    shape_0 = es.toGPUShape(bs.Shape(vertices_0, indices_0, None), GL_REPEAT, GL_LINEAR)
    shape_0.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_0.texture, "sprites/dark_wood_.png")
    shapes += [shape_0]

    #Suelo del acuario
    # deltas para los texeles
    dx = (W/min(W, D))*2
    dy = (D/min(W, D))*2
    vertices_1 = [
        0, 0, 0, 0, dy, 0, 0, 1,  # 0
        W, 0, 0, dx, dy, 0, 0, 1,  # 1
        W, D, 0, dx, 0, 0, 0, 1,  # 2
        0, D, 0, 0, 0, 0, 0, 1,  # 3
    ]
    indices_1 = [
        0, 1, 2, 2, 3, 0]
    shape_1 = es.toGPUShape(bs.Shape(vertices_1, indices_1, None), GL_REPEAT, GL_LINEAR)
    shape_1.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_1.texture, "sprites/gravel.png")
    shapes += [shape_1]

    max_side = max(W, D)
    dt = 4

    # Piso
    vertices_2 = [
        W/2 - (max_side/2 + rad), D/2 - (max_side/2 + rad), -sub_h, 0, dt, 0, 0, 1,  # 0
        W/2 + (max_side/2 + rad), D/2 - (max_side/2 + rad), -sub_h, dt, dt, 0, 0, 1,  # 1
        W/2 + (max_side/2 + rad), D/2 + (max_side/2 + rad), -sub_h, dt, 0, 0, 0, 1,  # 2
        W/2 - (max_side/2 + rad), D/2 + (max_side/2 + rad), -sub_h, 0, 0, 0, 0, 1   # 3
    ]
    indices_2 = [
        0, 1, 2, 2, 3, 0]
    shape_2 = es.toGPUShape(bs.Shape(vertices_2, indices_2, None), GL_REPEAT, GL_LINEAR)
    shape_2.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_2.texture, "sprites/floor.png")
    shapes += [shape_2]

    # Paredes
    dt = 4
    vertices_3 = [
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), -sub_h, 0, 0, 0, 1, 0,  # 0
        W / 2 - (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), -sub_h, 0, dt, 0, 1, 0,  # 1
        W / 2 - (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, dt, dt, 0, 1, 0,  # 2
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, dt, 0, 0, 1, 0,  # 3

        W / 2 - (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), -sub_h, 0, 0, 1, 0, 0,  # 4
        W / 2 - (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), -sub_h, 0, dt, 1, 0, 0,  # 5
        W / 2 - (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, dt, 1, 0, 0,  # 6
        W / 2 - (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, dt, 0, 1, 0, 0,  # 7

        W / 2 - (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), -sub_h, 0, 0, 0, -1, 0,  # 8
        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), -sub_h, 0, dt, 0, -1, 0,  # 9
        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, dt, 0, -1, 0,  # 10
        W / 2 - (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, 0, 0, -1, 0,  # 11
    ]
    indices_3 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8]
    shape_3 = es.toGPUShape(bs.Shape(vertices_3, indices_3, None), GL_REPEAT, GL_LINEAR)
    shape_3.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_3.texture, "sprites/wood.png")
    shapes += [shape_3]

    # Pared de la entrada
    tx_W = max_side + 2*rad
    tx_L = H + up_h + sub_h
    tx_x = (tx_W - D)/2
    tx_y = H + sub_h
    vertices_4 = [
        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), -sub_h, 0, 0, -1, 0, 0,  # 0
        W / 2 + (max_side / 2 + rad), D, -sub_h,                0, dt*tx_x/tx_W,  -1, 0, 0,  # 1
        W / 2 + (max_side / 2 + rad), D, H, dt*tx_y/tx_L, dt*tx_x/tx_W,   -1, 0, 0,  # 2
        W / 2 + (max_side / 2 + rad), D/2+(max_side/2+rad), H, dt*tx_y/tx_L, 0, -1, 0, 0,  # 3

        W / 2 + (max_side / 2 + rad), 0, -sub_h, 0, dt - dt*tx_x/tx_W, -1, 0, 0,  # 4
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), -sub_h, 0, dt, -1, 0, 0,  # 5
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H, dt*tx_y/tx_L, dt, -1, 0, 0,  # 6
        W / 2 + (max_side / 2 + rad), 0, H, dt*tx_y/tx_L, dt - dt * tx_x / tx_W, -1, 0, 0,  # 7

        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H, dt*tx_y/tx_L, 0, -1, 0, 0,  # 8
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H, dt*tx_y/tx_L, dt, -1, 0, 0, # 9
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, dt, dt, -1, 0, 0,  # 10
        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, 0, -1, 0, 0  # 11
    ]
    indices_4 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        3, 6, 10, 10, 11, 3]
    shape_4 = es.toGPUShape(bs.Shape(vertices_4, indices_4, None), GL_REPEAT, GL_LINEAR)
    shape_4.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_4.texture, "sprites/wood.png")
    shapes += [shape_4]

    # Techo
    vertices_5 = [
        W / 2 + (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, 0, 0, 0, 0, -1,  # 0
        W / 2 - (max_side / 2 + rad), D / 2 - (max_side / 2 + rad), H + up_h, 0, dt, 0, 0, -1, # 1
        W / 2 - (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, dt, 0, 0, -1,  # 2
        W / 2 + (max_side / 2 + rad), D / 2 + (max_side / 2 + rad), H + up_h, dt, 0, 0, 0, -1,  # 3
    ]
    indices_5 = [
        0, 1, 2, 2, 3, 0]
    shape_5 = es.toGPUShape(bs.Shape(vertices_5, indices_5, None), GL_REPEAT, GL_LINEAR)
    shape_5.texture = glGenTextures(1)
    textureWithMipMapSetup(shape_5.texture, "sprites/wood.png")
    shapes += [shape_5]

    # Entrada
    E = H
    g_S = 0.1
    vertices_6 = [
        W / 2 + (max_side / 2 + rad), D, -sub_h, g_S, g_S, g_S, 0, 0, 1,
        W / 2 + (max_side / 2 + rad), 0, -sub_h, g_S, g_S, g_S, 0, 0, 1,
        W / 2 + (max_side / 2 + rad) + E, 0, -sub_h, g_S, g_S, g_S, 0, 0, 1,
        W / 2 + (max_side / 2 + rad) + E, D, -sub_h, g_S, g_S, g_S, 0, 0, 1,

        W / 2 + (max_side / 2 + rad) + E, D, H, g_S, g_S, g_S, 0, 0, -1,
        W / 2 + (max_side / 2 + rad) + E, 0, H, g_S, g_S, g_S, 0, 0, -1,
        W / 2 + (max_side / 2 + rad), 0, H, g_S, g_S, g_S, 0, 0, -1,
        W / 2 + (max_side / 2 + rad), D, H, g_S, g_S, g_S, 0, 0, -1,

        W / 2 + (max_side / 2 + rad) + E, D, -sub_h, g_S, g_S, g_S, -1, 0, 0,
        W / 2 + (max_side / 2 + rad) + E, 0, -sub_h, g_S, g_S, g_S, -1, 0, 0,
        W / 2 + (max_side / 2 + rad) + E, 0, H, g_S, g_S, g_S, -1, 0, 0,
        W / 2 + (max_side / 2 + rad) + E, D, H, g_S, g_S, g_S, -1, 0, 0,

        W / 2 + (max_side / 2 + rad), D, -sub_h, g_S, g_S, g_S, 0, -1, 0,
        W / 2 + (max_side / 2 + rad) + E, D, -sub_h, g_S, g_S, g_S, 0, -1, 0,
        W / 2 + (max_side / 2 + rad) + E, D, H, g_S, g_S, g_S, 0, -1, 0,
        W / 2 + (max_side / 2 + rad), D, H, g_S, g_S, g_S, 0, -1, 0,

        W / 2 + (max_side / 2 + rad) + E, 0, -sub_h, g_S, g_S, g_S, 0, 1, 0,
        W / 2 + (max_side / 2 + rad), 0, -sub_h, g_S, g_S, g_S, 0, 1, 0,
        W / 2 + (max_side / 2 + rad), 0, H, g_S, g_S, g_S, 0, 1, 0,
        W / 2 + (max_side / 2 + rad) + E, 0, H, g_S, g_S, g_S, 0, 1, 0
    ]
    indices_6 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16
    ]
    shape_6 = bs.Shape(vertices_6, indices_6)
    shapes += [es.toGPUShape(shape_6)]

    #Reguladores termicos del acuario
    rH = 0.2
    #REGULADOR B
    cB = [0.3, 0.3, 0.3]
    rB_sw = [W/3, D/5]
    rB_se = [W*2/3, D/5]
    rB_ne = [W*2/3, D*2/5]
    rB_nw = [W/3, D*2/5]
    vertices_7 = [
        rB_sw[0], rB_sw[1], 0, cB[0], cB[1], cB[2], 0, -1, 0,  # 0
        rB_se[0], rB_se[1], 0, cB[0], cB[1], cB[2], 0, -1, 0,  # 1
        rB_se[0], rB_se[1], rH, cB[0], cB[1], cB[2], 0, -1, 0,  # 2
        rB_sw[0], rB_sw[1], rH, cB[0], cB[1], cB[2], 0, -1, 0,  # 3

        rB_se[0], rB_se[1], 0, cB[0], cB[1], cB[2], 1, 0, 0,  # 4
        rB_ne[0], rB_ne[1], 0, cB[0], cB[1], cB[2], 1, 0, 0,  # 5
        rB_ne[0], rB_ne[1], rH, cB[0], cB[1], cB[2], 1, 0, 0,  # 6
        rB_se[0], rB_se[1], rH,cB[0], cB[1], cB[2], 1, 0, 0,  # 7

        rB_ne[0], rB_ne[1], 0, cB[0], cB[1], cB[2], 0, 1, 0,  # 8
        rB_nw[0], rB_nw[1], 0, cB[0], cB[1], cB[2], 0, 1, 0,  # 9
        rB_nw[0], rB_nw[1], rH, cB[0], cB[1], cB[2], 0, 1, 0,  # 10
        rB_ne[0], rB_ne[1], rH, cB[0], cB[1], cB[2], 0, 1, 0,  # 11

        rB_nw[0], rB_nw[1], 0,cB[0], cB[1], cB[2], -1, 0, 0,  # 12
        rB_sw[0], rB_sw[1], 0, cB[0], cB[1], cB[2], -1, 0, 0,  # 13
        rB_sw[0], rB_sw[1], rH, cB[0], cB[1], cB[2], -1, 0, 0,  # 14
        rB_nw[0], rB_nw[1], rH, cB[0], cB[1], cB[2], -1, 0, 0,  # 15

        rB_sw[0], rB_sw[1], rH, cB[0], cB[1], cB[2], 0, 0, 1,  # 16
        rB_se[0], rB_se[1], rH, cB[0], cB[1], cB[2], 0, 0, 1,  # 17
        rB_ne[0], rB_ne[1], rH, cB[0], cB[1], cB[2], 0, 0, 1,  # 18
        rB_nw[0], rB_nw[1], rH, cB[0], cB[1], cB[2], 0, 0, 1,  # 19
    ]
    indices_7 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16
    ]
    shape_7 = bs.Shape(vertices_7, indices_7)
    shapes += [es.toGPUShape(shape_7)]

    # REGULADOR A
    cA = [0.3, 0.3, 0.3]
    rA_sw = [W / 3, D * 3 / 5]
    rA_se = [W * 2 / 3, D * 3 / 5]
    rA_ne = [W * 2 / 3, D * 4 / 5]
    rA_nw = [W / 3, D * 4 / 5]
    vertices_8 = [
        rA_sw[0], rA_sw[1], 0, cA[0], cA[1], cA[2], 0, -1, 0,  # 0
        rA_se[0], rA_se[1], 0, cA[0], cA[1], cA[2], 0, -1, 0,  # 1
        rA_se[0], rA_se[1], rH, cA[0], cA[1], cA[2], 0, -1, 0,  # 2
        rA_sw[0], rA_sw[1], rH, cA[0], cA[1], cA[2], 0, -1, 0,  # 3

        rA_se[0], rA_se[1], 0, cA[0], cA[1], cA[2], 1, 0, 0,  # 4
        rA_ne[0], rA_ne[1], 0, cA[0], cA[1], cA[2], 1, 0, 0,  # 5
        rA_ne[0], rA_ne[1], rH, cA[0], cA[1], cA[2], 1, 0, 0,  # 6
        rA_se[0], rA_se[1], rH, cA[0], cA[1], cA[2], 1, 0, 0,  # 7

        rA_ne[0], rA_ne[1], 0, cA[0], cA[1], cA[2], 0, 1, 0,  # 8
        rA_nw[0], rA_nw[1], 0, cA[0], cA[1], cA[2], 0, 1, 0,  # 9
        rA_nw[0], rA_nw[1], rH, cA[0], cA[1], cA[2], 0, 1, 0,  # 10
        rA_ne[0], rA_ne[1], rH, cA[0], cA[1], cA[2], 0, 1, 0,  # 11

        rA_nw[0], rA_nw[1], 0, cA[0], cA[1], cA[2], -1, 0, 0,  # 12
        rA_sw[0], rA_sw[1], 0, cA[0], cA[1], cA[2], -1, 0, 0,  # 13
        rA_sw[0], rA_sw[1], rH, cA[0], cA[1], cA[2], -1, 0, 0,  # 14
        rA_nw[0], rA_nw[1], rH, cA[0], cA[1], cA[2], -1, 0, 0,  # 15

        rA_sw[0], rA_sw[1], rH, cA[0], cA[1], cA[2], 0, 0, 1,  # 16
        rA_se[0], rA_se[1], rH, cA[0], cA[1], cA[2], 0, 0, 1,  # 17
        rA_ne[0], rA_ne[1], rH, cA[0], cA[1], cA[2], 0, 0, 1,  # 18
        rA_nw[0], rA_nw[1], rH, cA[0], cA[1], cA[2], 0, 0, 1,  # 19
    ]
    indices_8 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16
    ]
    shape_8 = bs.Shape(vertices_8, indices_8)
    shapes += [es.toGPUShape(shape_8)]

    min_side = min(W, H)
    height = 1.34 * min_side
    wall_dist = 0.5
    #Squirtle Poster
    vertices_9 = [
        W / 2 + min_side/2, D / 2 - (max_side / 2 + rad) + wall_dist, H/2 -height/2, 0, 1, 0, 1, 0,  # 0
        W / 2 - min_side/2, D / 2 - (max_side / 2 + rad) + wall_dist, H/2 -height/2, 1, 1, 0, 1, 0,  # 1
        W / 2 - min_side/2, D / 2 - (max_side / 2 + rad) + wall_dist, H/2 +height/2, 1, 0, 0, 1, 0,  # 2
        W / 2 + min_side/2, D / 2 - (max_side / 2 + rad) + wall_dist, H/2 +height/2, 0, 0, 0, 1, 0,  # 3
    ]
    indices_9 = [
        0, 1, 2, 2, 3, 0]
    shape_9 = es.toGPUShape(bs.Shape(vertices_9, indices_9, "sprites/squirtle_poster.png"), GL_REPEAT, GL_LINEAR)
    shapes += [shape_9]

    height = 1.5 * min_side
    # Magikarp Poster
    vertices_10 = [
        W / 2 - (max_side / 2 + rad) + wall_dist, D / 2 - min_side / 2, H / 2 - height / 2, 0, 1, 1, 0, 0,  # 4
        W / 2 - (max_side / 2 + rad) + wall_dist, D / 2 + min_side / 2, H / 2 - height / 2, 1, 1, 1, 0, 0,  # 5
        W / 2 - (max_side / 2 + rad) + wall_dist, D / 2 + min_side / 2, H / 2 + height / 2, 1, 0, 1, 0, 0,  # 6
        W / 2 - (max_side / 2 + rad) + wall_dist, D / 2 - min_side / 2, H / 2 + height / 2, 0, 0, 1, 0, 0,  # 7
    ]
    indices_10 = [
        0, 1, 2, 2, 3, 0]
    shape_10 = es.toGPUShape(bs.Shape(vertices_10, indices_10, "sprites/magikarp_poster.png"), GL_REPEAT, GL_LINEAR)
    shapes += [shape_10]

    height = 1.41 * min_side
    # Squirtle Poster
    vertices_11 = [
        W / 2 - min_side / 2, D / 2 + (max_side / 2 + rad) - wall_dist, H / 2 - height / 2, 0, 1, 0, -1, 0,  # 0
        W / 2 + min_side / 2, D / 2 + (max_side / 2 + rad) - wall_dist, H / 2 - height / 2, 1, 1, 0, -1, 0,  # 1
        W / 2 + min_side / 2, D / 2 + (max_side / 2 + rad) - wall_dist, H / 2 + height / 2, 1, 0, 0, -1, 0,  # 2
        W / 2 - min_side / 2, D / 2 + (max_side / 2 + rad) - wall_dist, H / 2 + height / 2, 0, 0, 0, -1, 0,  # 3
    ]
    indices_11 = [
        0, 1, 2, 2, 3, 0 ]
    shape_11 = es.toGPUShape(bs.Shape(vertices_11, indices_11, "sprites/vaporeon_poster.png"), GL_REPEAT, GL_LINEAR)
    shapes += [shape_11]

    return shapes

#Funcion para crear los elementos del acuario
def create_aquarium(dim, side_b, up_b, up_gap, offset):
    W, D, H = dim[0], dim[1], dim[2]
    shapes = []

    # CORNER POINTS
    sw_down = [0 - offset, 0 - offset, 0]
    se_down = [W + offset, 0 - offset, 0]
    nw_down = [0 - offset, D + offset, 0]
    ne_down = [W + offset, D + offset, 0]
    sw_up = [0 - offset, 0 - offset, H + up_gap]
    se_up = [W + offset, 0 - offset, H + up_gap]
    nw_up = [0 - offset, D + offset, H + up_gap]
    ne_up = [W + offset, D + offset, H + up_gap]

    #Bordes laterales
    s_c = [0.8, 0.8, 0.8]
    vertices_0 = [
        sw_down[0], sw_down[1], sw_down[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        sw_down[0] + side_b, sw_down[1], sw_down[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        sw_up[0] + side_b, sw_up[1], sw_up[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        sw_up[0], sw_up[1], sw_up[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,

        se_down[0] - side_b, se_down[1], se_down[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        se_down[0], se_down[1], se_down[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        se_up[0], se_up[1], se_up[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,
        se_up[0] - side_b, se_up[1], se_up[2], s_c[0], s_c[1], s_c[2], 0, -1, 0,

        ne_down[0], ne_down[1], ne_down[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        ne_down[0] - side_b, ne_down[1], ne_down[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        ne_up[0] - side_b, ne_up[1], ne_up[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        ne_up[0], ne_up[1], ne_up[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,

        nw_down[0] + side_b, nw_down[1], nw_down[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        nw_down[0], nw_down[1], nw_down[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        nw_up[0], nw_up[1], nw_up[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,
        nw_up[0] + side_b, nw_up[1], nw_up[2], s_c[0], s_c[1], s_c[2], 0, 1, 0,

        se_down[0], se_down[1], se_down[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        se_down[0], se_down[1] + side_b, se_down[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        se_up[0], se_up[1] + side_b, se_up[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        se_up[0], se_up[1], se_up[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,

        ne_down[0], ne_down[1] - side_b, ne_down[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        ne_down[0], ne_down[1], ne_down[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        ne_up[0], ne_up[1], ne_up[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,
        ne_up[0], ne_up[1] - side_b, ne_up[2], s_c[0], s_c[1], s_c[2], 1, 0, 0,

        nw_down[0], nw_down[1], nw_down[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        nw_down[0], nw_down[1] - side_b, nw_down[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        nw_up[0], nw_up[1] - side_b, nw_up[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        nw_up[0], nw_up[1], nw_up[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,

        sw_down[0], sw_down[1] + side_b, sw_down[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        sw_down[0], sw_down[1], sw_down[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        sw_up[0], sw_up[1], sw_up[2], s_c[0], s_c[1], s_c[2], -1, 0, 0,
        sw_up[0], sw_up[1] + side_b, sw_up[2], s_c[0], s_c[1], s_c[2], -1, 0, 0
    ]
    indices_0 =[
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20,
        24, 25, 26, 26, 27, 24,
        28, 29, 30, 30, 31, 28
    ]
    shape_0 = bs.Shape(vertices_0, indices_0)
    shapes += [es.toGPUShape(shape_0)]

    #Bordes superiores
    up_c = [0.2, 0.2, 0.2]
    vertices_1 = [
        sw_up[0], sw_up[1], sw_up[2], up_c[0], up_c[1], up_c[2], 0, -1, 0,
        se_up[0], se_up[1], se_up[2], up_c[0], up_c[1], up_c[2], 0, -1, 0,
        se_up[0], se_up[1], se_up[2] + up_b, up_c[0], up_c[1], up_c[2], 0, -1, 0,
        sw_up[0], sw_up[1], sw_up[2] + up_b, up_c[0], up_c[1], up_c[2], 0, -1, 0,

        ne_up[0], ne_up[1], ne_up[2], up_c[0], up_c[1], up_c[2], 0, 1, 0,
        nw_up[0], nw_up[1], nw_up[2], up_c[0], up_c[1], up_c[2], 0, 1, 0,
        nw_up[0], nw_up[1], nw_up[2] + up_b, up_c[0], up_c[1], up_c[2], 0, 1, 0,
        ne_up[0], ne_up[1], ne_up[2] + up_b, up_c[0], up_c[1], up_c[2], 0, 1, 0,

        se_up[0], se_up[1], se_up[2], up_c[0], up_c[1], up_c[2], 1, 0, 0,
        ne_up[0], ne_up[1], ne_up[2], up_c[0], up_c[1], up_c[2], 1, 0, 0,
        ne_up[0], ne_up[1], ne_up[2] + up_b, up_c[0], up_c[1], up_c[2], 1, 0, 0,
        se_up[0], se_up[1], se_up[2] + up_b, up_c[0], up_c[1], up_c[2], 1, 0, 0,

        nw_up[0], nw_up[1], nw_up[2], up_c[0], up_c[1], up_c[2], -1, 0, 0,
        sw_up[0], sw_up[1], sw_up[2], up_c[0], up_c[1], up_c[2], -1, 0, 0,
        sw_up[0], sw_up[1], sw_up[2] + up_b, up_c[0], up_c[1], up_c[2], -1, 0, 0,
        nw_up[0], nw_up[1], nw_up[2] + up_b, up_c[0], up_c[1], up_c[2], -1, 0, 0
    ]
    indices_1 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12
    ]
    shape_1 = bs.Shape(vertices_1, indices_1)
    shapes += [es.toGPUShape(shape_1)]

    # Vidrios transparentes
    up_c = [56/255, 157/255, 235/255]
    vertices_2 = [
        sw_down[0] + side_b, sw_down[1], sw_down[2], up_c[0], up_c[1], up_c[2],  0, 1, 0,
        se_down[0] - side_b, se_down[1], se_down[2], up_c[0], up_c[1], up_c[2],  0, 1, 0,
        se_down[0] - side_b, se_down[1], se_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 1, 0,
        sw_down[0] + side_b, sw_down[1], sw_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 1, 0,

        ne_down[0] - side_b, ne_down[1], ne_down[2], up_c[0], up_c[1], up_c[2],  0, -1, 0,
        nw_down[0] + side_b, nw_down[1], nw_down[2],  up_c[0], up_c[1], up_c[2], 0, -1, 0,
        nw_down[0] + side_b, nw_down[1], nw_down[2] + H, up_c[0], up_c[1], up_c[2], 0, -1, 0,
        ne_down[0] - side_b, ne_down[1], ne_down[2] + H, up_c[0], up_c[1], up_c[2],  0, -1, 0,

        se_down[0], se_down[1] + side_b, se_down[2], up_c[0], up_c[1], up_c[2],  -1, 0, 0,
        ne_down[0], ne_down[1] - side_b, ne_down[2], up_c[0], up_c[1], up_c[2],  -1, 0, 0,
        ne_down[0], ne_down[1] - side_b, ne_down[2] + H, up_c[0], up_c[1], up_c[2],  -1, 0, 0,
        se_down[0], se_down[1] + side_b, se_down[2] + H, up_c[0], up_c[1], up_c[2],  -1, 0, 0,

        nw_down[0], nw_down[1] - side_b, nw_down[2], up_c[0], up_c[1], up_c[2],  1, 0, 0,
        sw_down[0], sw_down[1] + side_b, sw_down[2], up_c[0], up_c[1], up_c[2],  1, 0, 0,
        sw_down[0], sw_down[1] + side_b, sw_down[2] + H, up_c[0], up_c[1], up_c[2],  1, 0, 0,
        nw_down[0], nw_down[1] - side_b, nw_down[2] + H, up_c[0], up_c[1], up_c[2],  1, 0, 0,

        se_down[0], se_down[1], se_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 0, -1,
        ne_down[0], ne_down[1], ne_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 0, -1,
        nw_down[0], nw_down[1], nw_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 0, -1,
        sw_down[0], sw_down[1], sw_down[2] + H, up_c[0], up_c[1], up_c[2],  0, 0, - 1
    ]
    indices_2 = [
        0, 3, 2, 2, 1, 0,
        4, 7, 6, 6, 5, 4,
        8, 11, 10, 10, 9, 8,
        12, 15, 14, 14, 13, 12,
        16, 19, 18, 18, 17, 16
    ]

    shape_2 = bs.Shape(vertices_2, indices_2)
    shapes += [es.toGPUShape(shape_2)]

    return shapes

#Funcion para crear una luz led
def create_led(c1, c2, sides, h=1):
    angle = 2*np.pi/sides
    ring = 0.7

    vertices_0 =[0, 0, -h/2, c1[0], c1[1], c1[2], 0 ,0 ,-1]
    indices_0=[]


    for i in range(sides+1):
        vertices_0 += [np.cos(angle*i), np.sin(angle*i), h/2, c2[0], c2[1], c2[2], np.cos(angle*i), np.sin(angle*i), 0]
        vertices_0 += [np.cos(angle * i), np.sin(angle * i), -h / 2, c2[0], c2[1], c2[2], np.cos(angle * i),
                       np.sin(angle * i), 0]
        vertices_0 += [np.cos(angle * i), np.sin(angle * i), -h / 2, c2[0], c2[1], c2[2], 0, 0, -1]

        vertices_0 += [ring*np.cos(angle * i), ring*np.sin(angle * i), -h / 2, c2[0], c2[1], c2[2], 0, 0, -1]
        vertices_0 += [ring*np.cos(angle * i), ring*np.sin(angle * i), -h / 2, c1[0], c1[1], c1[2], 0, 0, -1]
        if i < sides:
            indices_0 += [5*i + 1, 5*i + 2, 5*i + 7, 5*i + 7, 5*i + 6, 5*i + 1]

            indices_0 += [5*i + 3, 5*i + 4, 5*i + 9, 5*i + 9, 5*i + 8, 5*i + 3]
            indices_0 += [5*i + 5, 0,5*i+10]

    return bs.Shape(vertices_0, indices_0)