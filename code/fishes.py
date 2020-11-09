import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import math

import transformations as tr
import easy_shaders as es
import basic_shapes as bs
import lighting_shaders as ls

import scene_effects as fx

#Pipeline para los peces
LightPipeline = None
#Lista que contendra los peces de la escena
Fishes = []

#Objeto pez
class Fish_Object:
    def __init__(self, pos, gpu_shape, gpu_tail):
        self.position = pos #posicion inicial
        self.shape = gpu_shape #shape del cuerpo
        self.tail_shape = gpu_tail #shape de la cola
        self.transform_0 = None #transformaciones para encajar la cola en el origen
        self.transform_1 = None #transformaciones para colocar la cola rotada en el cuerpo
        self.tr_body = None #transformaciones del cuerpo para encajar la cola
        self.tail_rot = 0 #rotacion de la cola
        self.tail_speed = 1 #velocidad de rotacion de la cola
        self.rot_range = 1 #Amplitud de movimiento de la cola
        self.rot_offset = 0 #offset del movimiento de la cola
        self.axis_rot = 0 # 0->Y, 1->Z, eje en que se mueve la cola
        self.pipeline = None #Pipeline con el que se dibujarqa

        self.in_voxels = None #matriz del acuario con 1s en los voxeles dentro del rango
        self.direction = np.array([-1,0,0]) #direccion nde movimiento
        self.target_pos = np.array([0,0,0]) #posicion del objetivo
        self.velocity = 2 #velocidad de movimiento
        self.dir_counter = 0 #contador de movimiento en la misma direccion
        self.dir_amount = 5 #maxima cantidad de movimiento en la misma direccion
        self.movement = 0 #cantidad de movimiento

        self.is_rotating = False #parametro que indica si se esta rotanto a una nueva direccion
        self.rotation = 0 #rotacion actual (polar)
        self.target_rot = 0 #rotacion a alcanzar
        self.rot_speed = 2 # velocidad de rotacion

    #Metodo para dibujar al pez
    def draw(self):
        #Transformacion del cuerpo (posicion + rotacion + transformacion base)
        body_transform = tr.matmul([
            tr.translate(self.position[0], self.position[1], self.position[2]),
            tr.rotationZ(self.rotation),
            self.tr_body
        ])
        #Se dibuja el cuerpo
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "model"), 1, GL_TRUE, body_transform)
        self.pipeline.drawShape(self.shape)

        #La cola rota en el eje Y
        if self.axis_rot == 0:
            # Transformacion de la cola (posicion + rotacion del pez + tr base 1 + rotacion cola + tr base 0)
            transform = tr.matmul([
                tr.translate(self.position[0], self.position[1], self.position[2]),
                tr.rotationZ(self.rotation),
                self.transform_1,
                tr.rotationY(self.rot_range * np.sin(self.rot_offset + self.tail_rot)),
                self.transform_0
            ])
        # La cola rota en el eje Z
        else:
            # Transformacion de la cola (posicion + rotacion del pez + tr base 1 + rotacion cola + tr base 0)
            transform = tr.matmul([
                tr.translate(self.position[0], self.position[1], self.position[2]),
                tr.rotationZ(self.rotation),
                self.transform_1,
                tr.rotationZ(self.rot_range * np.sin(self.rot_offset + self.tail_rot)),
                self.transform_0
            ])
        #Se dibuja la cola
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "model"), 1, GL_TRUE, transform)
        self.pipeline.drawShape(self.tail_shape)

    #Metodo para actualizar la direccion de movimiento
    def target(self):
        #dimension del aquario
        dim = self.in_voxels.shape

        #posicion tentativa (se trata de mantener la misma direccion)
        tent_pos = self.position + self.direction
        if tent_pos[0] > 1 and tent_pos[0] < dim[0] and \
                tent_pos[1] > 1 and tent_pos[1] < dim[1] and \
                tent_pos[2] > 1 and tent_pos[2] < dim[2] and \
                self.dir_counter < self.dir_amount and \
                self.in_voxels[int(tent_pos[0] - 0.5), int(tent_pos[1] - 0.5), int(tent_pos[2] - 0.5)] == 1:
            # ni se ha alcanzado la maxima cantidad de movimiento por direccion
            # ni se ha salido de la zona: se mantiene moviendo en la misma direccion
            self.target_pos = tent_pos
            self.dir_counter += 1
        else:
            #De lo contrario se elige aleatoriamente un vexel vecino hacia el cual moverse
            in_range = False
            while not in_range:
                rand_index = np.random.randint(6)
                direction = np.array([0, 0, 0])
                if rand_index == 0:
                    direction = np.array([1, 0, 0])
                elif rand_index == 1:
                    direction = np.array([-1, 0, 0])
                elif rand_index == 2:
                    direction = np.array([0, 1, 0])
                elif rand_index == 3:
                    direction = np.array([0, -1, 0])
                elif rand_index == 4:
                    direction = np.array([0, 0, 1])
                elif rand_index == 5:
                    direction = np.array([0, 0, -1])

                tent_pos = self.position + direction
                if tent_pos[0] > 1 and tent_pos[0] < dim[0] and \
                        tent_pos[1] > 1 and tent_pos[1] < dim[1] and \
                        tent_pos[2] > 1 and tent_pos[2] < dim[2]:
                    if self.in_voxels[int(tent_pos[0] - 0.5), int(tent_pos[1] - 0.5), int(tent_pos[2] - 0.5)] == 1:
                        self.target_pos = tent_pos
                        self.direction = self.target_pos - self.position
                        in_range = True
                        self.dir_counter = 0
                        if self.position[2] == self.target_pos[2]:
                            self.target_rot = AngleBetween(self.position, self.target_pos)
                            self.is_rotating = True
    def update(self, delta):
        #dibuja el pez
        self.draw()
        #Mueve la cola un delta theta
        self.tail_rot = (self.tail_rot + self.tail_speed*delta)%(2*np.pi)

        #Mientras no termine de girar hacia la nueva direccion, no se mueve el pez
        if self.is_rotating:
            if np.abs(self.rotation - self.target_rot) < 0.1:
                self.rotation = self.target_rot
                self.is_rotating = False
            else:
                self.rotation = (self.rotation + self.rot_speed*delta)%(2*np.pi)
        else:
            if self.movement >= 1:
                #Si se ha movido una unidad/voxel, se procesa para ver si cambia de  direccion
                self.movement = 0
                self.position = self.target_pos
                self.target()
            else:
                #se mueve segun su direccion
                self.position += self.direction * delta
                self.movement += 1*delta

#Funcion que retorna el angulo entre dos vectores
def AngleBetween(pos1, pos2):
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    angle = 0
    if x > 0 and y >= 0:
        angle = np.arctan(y/x)
    elif x == 0 and y > 0:
        angle = np.pi/2
    elif x < 0:
        angle = np.arctan(y/x) + np.pi
    elif x == 0 and y < 0:
        angle = 3*np.pi/2
    elif x > 0 and y < 0:
        angle = np.arctan(y/x) + 2*np.pi
    angle = (angle + np.pi)%(2*np.pi)
    return angle

#Funciones para leer un archivo obj
def readFaceVertex(faceDescription):

    aux = faceDescription.split('/')

    assert len(aux[0]), "Vertex index has not been defined."

    faceVertex = [int(aux[0]), None, None]

    assert len(aux) == 3, "Only faces where its vertices require 3 indices are defined."

    if len(aux[1]) != 0:
        faceVertex[1] = int(aux[1])

    if len(aux[2]) != 0:
        faceVertex[2] = int(aux[2])

    return faceVertex
def readOBJ(filename, image_file_name, light):

    vertices = []
    normals = []
    textCoords= []
    faces = []

    with open(filename, 'r') as file:
        for line in file.readlines():
            aux = line.strip().split(' ')

            if aux[0] == 'v':
                vertices += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vn':
                normals += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vt':
                assert len(aux[1:]) == 2, "Texture coordinates with different than 2 dimensions are not supported"
                textCoords += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'f':
                N = len(aux)
                faces += [[readFaceVertex(faceVertex) for faceVertex in aux[1:4]]]
                for i in range(3, N- 1):
                    faces += [[readFaceVertex(faceVertex) for faceVertex in [aux[i], aux[i + 1], aux[1]]]]

        vertexData = []
        indices = []
        index = 0

        if light:
            # Per previous construction, each face is a triangle
            for face in faces:

                # Checking each of the triangle vertices
                for i in range(0, 3):
                    vertex = vertices[face[i][0] - 1]
                    texture = textCoords[face[i][1] - 1]
                    normal = normals[face[i][2] - 1]

                    vertexData += [
                        vertex[0], vertex[1], vertex[2],
                        texture[0], 1-texture[1],
                        normal[0], normal[1], normal[2]
                    ]

                # Connecting the 3 vertices to create a triangle
                indices += [index, index + 1, index + 2]
                index += 3
        else:
            # Per previous construction, each face is a triangle
            for face in faces:

                # Checking each of the triangle vertices
                for i in range(0, 3):
                    vertex = vertices[face[i][0] - 1]
                    texture = textCoords[face[i][1] - 1]

                    vertexData += [
                        vertex[0], vertex[1], vertex[2],
                        texture[0], 1-texture[1]
                    ]

                # Connecting the 3 vertices to create a triangle
                indices += [index, index + 1, index + 2]
                index += 3

        return bs.Shape(vertexData, indices, image_file_name)

#Funcion pra encontrar un float aleatorio entre dos numeros
def Random_between(n1, n2):
    len = n2 - n1
    return n1 + len*np.random.random()

#Se crean los peces: Vaporeon
def Vaporeon_Setup(voxels, amount, collection, dims, in_voxels):
    #Si no hay voxeles no se pueden crear peces
    if voxels == []:
        return
    #Se cargan los modelos de la cola y el cuerpo
    fish = es.toGPUShape(readOBJ("models/vaporeon.obj", "models/vaporeon.png", light=True), GL_CLAMP_TO_EDGE,
                           GL_LINEAR)
    tail = es.toGPUShape(readOBJ("models/vaporeon_tail.obj", "models/vaporeon_tail.png", light=True),
                           GL_CLAMP_TO_EDGE, GL_LINEAR)
    #Transformaciones del cuerpo
    tr_body = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.2, 0, 0),
        tr.uniformScale(0.03)
    ])
    #Transformaciones de la cola, antes del movimiento de la cola
    tr_tail0 = tr.matmul([
        tr.translate(0.1725, 0, -0.055),
        tr.uniformScale(0.03)
    ])
    # Transformaciones de la cola, despues del movimiento de la cola
    tr_tail1 = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.2, 0, 0),
        tr.translate(0.14, 0, 0.017)
    ])
    #valor para calcular la velocidad de movimiento
    movement = (dims[0]+dims[1]+dims[2])/3
    count = 0
    spawn_voxels = voxels[:]
    #Se crean los peces
    while count < amount:
        #Si se han puesto peces en todos los voxeles, se empiezan a repetir peces
        if spawn_voxels == []:
            spawn_voxels = voxels[:]
        # se escoge una posicion aleatoria de la zona
        temp_random = np.random.randint(len(spawn_voxels))
        temp_pos = spawn_voxels.pop(temp_random)
        #si esta posicion no esta en los bordes, sirve para colocar un pez
        if temp_pos[0] > 1 and temp_pos[0] < dims[0] and \
                temp_pos[1] > 1 and temp_pos[1] < dims[1] and \
                temp_pos[2] > 1 :
            #Se asignan los valores para el movimiento del pez
            temp_fish = Fish_Object(temp_pos, fish, tail)
            temp_fish.transform_0 = tr_tail0
            temp_fish.transform_1 = tr_tail1
            temp_fish.tr_body = tr_body
            temp_fish.tail_speed = Random_between(1,3)
            temp_fish.velocity = Random_between(movement/23,movement/7)
            temp_fish.rot_speed = Random_between(0.5, 2)
            temp_fish.tail_rot = Random_between(0.0, 2*np.pi)
            temp_fish.rot_range = 0.7
            temp_fish.rot_offset = -1.2
            temp_fish.dir_amount = np.random.randint(2,6)
            temp_fish.dir_counter = temp_fish.dir_amount
            temp_fish.pipeline = LightPipeline
            temp_fish.in_voxels = in_voxels
            temp_fish.target()
            collection += [temp_fish]
            count += 1

#Se crean los peces: Squirtle
def Squirtle_Setup(voxels, amount, collection, dims, in_voxels):
    # Si no hay voxeles no se pueden crear peces
    if voxels == []:
        return
    # Se cargan los modelos de la cola y el cuerpo
    fish = es.toGPUShape(readOBJ("models/squirtle.obj", "models/squirtle.png", light=True), GL_CLAMP_TO_EDGE,
                           GL_LINEAR)
    tail = es.toGPUShape(readOBJ("models/squirtle_tail.obj", "models/squirtle_tail.png", light=True),
                           GL_CLAMP_TO_EDGE, GL_LINEAR)
    # Transformaciones del cuerpo
    tr_body = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.1, 0, 0),
        tr.uniformScale(0.01)
    ])
    # Transformaciones de la cola, antes del movimiento de la cola
    tr_tail0 = tr.matmul([
        tr.translate(0.1, 0, -0.093),
        tr.uniformScale(0.01)
    ])
    # Transformaciones de la cola, despues del movimiento de la cola
    tr_tail1 = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.1, 0, 0),
        tr.translate(0.20, 0, -0.045),
    ])
    # valor para calcular la velocidad de movimiento
    movement = (dims[0] + dims[1] + dims[2]) / 3
    count = 0
    spawn_voxels = voxels[:]
    # Se crean los peces
    while count < amount:
        # Si se han puesto peces en todos los voxeles, se empiezan a repetir peces
        if spawn_voxels == []:
            spawn_voxels = voxels[:]
        # se escoge una posicion aleatoria de la zona
        temp_random = np.random.randint(len(spawn_voxels))
        temp_pos = spawn_voxels.pop(temp_random)
        # si esta posicion no esta en los bordes, sirve para colocar un pez
        if temp_pos[0] > 1 and temp_pos[0] < dims[0] and \
                temp_pos[1] > 1 and temp_pos[1] < dims[1] and \
                temp_pos[2] > 1:
            # Se asignan los valores para el movimiento del pez
            temp_fish = Fish_Object(temp_pos, fish, tail)
            temp_fish.transform_0 = tr_tail0
            temp_fish.transform_1 = tr_tail1
            temp_fish.tr_body = tr_body
            temp_fish.tail_speed = Random_between(1, 3)
            temp_fish.tail_rot = Random_between(0.0, 2*np.pi)
            temp_fish.velocity = Random_between(movement/23,movement/7)
            temp_fish.rot_speed = Random_between(0.5, 2)
            temp_fish.dir_amount = np.random.randint(2, 6)
            temp_fish.dir_counter = temp_fish.dir_amount
            temp_fish.axis_rot = 1
            temp_fish.pipeline = LightPipeline
            temp_fish.in_voxels = in_voxels
            temp_fish.target()
            collection += [temp_fish]
            count += 1

#Se crean los peces: Magikarp
def Magikarp_Setup(voxels, amount, collection, dims, in_voxels):
    # Si no hay voxeles no se pueden crear peces
    if voxels == []:
        return
    # Se cargan los modelos de la cola y el cuerpo
    fish = es.toGPUShape(readOBJ("models/magikarp.obj", "models/magikarp.png", light=True), GL_CLAMP_TO_EDGE,
                           GL_LINEAR)
    tail = es.toGPUShape(readOBJ("models/magikarp_tail.obj", "models/magikarp_tail.png", light=True),
                           GL_CLAMP_TO_EDGE, GL_LINEAR)
    # Transformaciones del cuerpo
    tr_body = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.07, 0, 0),
        tr.uniformScale(0.0075)
    ])
    # Transformaciones de la cola, antes del movimiento de la cola
    tr_tail0 = tr.matmul([
        tr.translate(0.092, 0, 0),
        tr.uniformScale(0.0075)
    ])
    # Transformaciones de la cola, despues del movimiento de la cola
    tr_tail1 = tr.matmul([
        tr.uniformScale(5),
        tr.translate(-0.07, 0, 0),
        tr.translate(0.15, 0, 0),
    ])
    # valor para calcular la velocidad de movimiento
    movement = (dims[0] + dims[1] + dims[2]) / 3
    count = 0
    spawn_voxels = voxels[:]
    # Se crean los peces
    while count < amount:
        # Si se han puesto peces en todos los voxeles, se empiezan a repetir peces
        if spawn_voxels == []:
            spawn_voxels = voxels[:]
        # se escoge una posicion aleatoria de la zona
        temp_random = np.random.randint(len(spawn_voxels))
        temp_pos = spawn_voxels.pop(temp_random)
        # si esta posicion no esta en los bordes, sirve para colocar un pez
        if temp_pos[0] > 1 and temp_pos[0] < dims[0] and \
                temp_pos[1] > 1 and temp_pos[1] < dims[1] and \
                temp_pos[2] > 1:
            # Se asignan los valores para el movimiento del pez
            temp_fish = Fish_Object(temp_pos, fish, tail)
            temp_fish.transform_0 = tr_tail0
            temp_fish.transform_1 = tr_tail1
            temp_fish.tr_body = tr_body
            temp_fish.tail_speed = Random_between(1, 3)
            temp_fish.tail_rot = Random_between(0.0, 2*np.pi)
            temp_fish.velocity = Random_between(movement/23,movement/7)
            temp_fish.rot_speed = Random_between(0.5, 2)
            temp_fish.dir_amount = np.random.randint(2, 6)
            temp_fish.dir_counter = temp_fish.dir_amount
            temp_fish.axis_rot = 1
            temp_fish.pipeline = LightPipeline
            temp_fish.in_voxels = in_voxels
            temp_fish.target()
            collection += [temp_fish]
            count += 1

#Funcion para inicializar los peces
def Setup(a_pos, b_pos, c_pos, n_a, n_b, n_c, dims, a_voxels, b_voxels, c_voxels):
    global LightPipeline, Fishes

    #Se asigna el pipeline
    LightPipeline = fx.FourTexturePhongShaderProgram()
    #Se generan los peces
    Vaporeon_Setup(a_pos, n_a, Fishes, dims, a_voxels)
    Squirtle_Setup(b_pos, n_b, Fishes, dims, b_voxels)
    Magikarp_Setup(c_pos, n_c, Fishes, dims, c_voxels)

#Funcion para actualizar los peces
def Update(delta):
    #MAterial de los peces
    glUniform3f(glGetUniformLocation(LightPipeline.shaderProgram, "Ka"), 0.4, 0.4, 0.4)
    glUniform3f(glGetUniformLocation(LightPipeline.shaderProgram, "Kd"), 0.6, 0.6, 0.6)
    glUniform3f(glGetUniformLocation(LightPipeline.shaderProgram, "Ks"), 0.9, 0.9, 0.9)
    #Se dibuja cada pez
    for fish in Fishes:
        fish.update(delta)