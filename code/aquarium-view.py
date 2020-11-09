# Sebastian Olmos

import json
import glfw
from glfw.GLFW import *
from OpenGL.GL import *
import numpy as np
import sys

import transformations as tr

import user_controller as user
import scene_manager as scene

#Cantidad de frames
FPS = 60
#Dimensiones de la ventana, proporcion ideal de 16/9
WIDTH = 1440
HEIGHT = 810

def on_key(window, key, scancode, action, mods):

    if key == glfw.KEY_SPACE:
        if action == glfw.PRESS:
            controller.fillPolygon = not controller.fillPolygon

    if key == glfw.KEY_ESCAPE:
        if action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    elif key == glfw.KEY_LEFT_CONTROL:
        if action == glfw.PRESS:
            controller.is_ctrl_pressed = not controller.is_ctrl_pressed

    elif key == glfw.KEY_V:
        if action == glfw.PRESS:
            controller.fp_view = not controller.fp_view

    if key == glfw.KEY_UP:
        if action == glfw.PRESS:
            controller.is_up_pressed = True
        elif action == glfw.RELEASE:
            controller.is_up_pressed = False

    if key == glfw.KEY_DOWN:
        if action == glfw.PRESS:
            controller.is_down_pressed = True
        elif action == glfw.RELEASE:
            controller.is_down_pressed = False

    if key == glfw.KEY_RIGHT:
        if action == glfw.PRESS:
            controller.is_right_pressed = True
        elif action == glfw.RELEASE:
            controller.is_right_pressed = False

    if key == glfw.KEY_LEFT:
        if action == glfw.PRESS:
            controller.is_left_pressed = True
        elif action == glfw.RELEASE:
            controller.is_left_pressed = False

    if key == glfw.KEY_Z:
        if action == glfw.PRESS:
            controller.is_z_pressed = True
        elif action == glfw.RELEASE:
            controller.is_z_pressed = False

    if key == glfw.KEY_X:
        if action == glfw.PRESS:
            controller.is_x_pressed = True
        elif action == glfw.RELEASE:
            controller.is_x_pressed = False

    if key == glfw.KEY_E:
        if action == glfw.PRESS:
            controller.is_e_pressed = True
        elif action == glfw.RELEASE:
            controller.is_e_pressed = False

    if key == glfw.KEY_S:
        if action == glfw.PRESS:
            controller.is_s_pressed = True
        elif action == glfw.RELEASE:
            controller.is_s_pressed = False

    if key == glfw.KEY_F:
        if action == glfw.PRESS:
            controller.is_f_pressed = True
        elif action == glfw.RELEASE:
            controller.is_f_pressed = False

    if key == glfw.KEY_D:
        if action == glfw.PRESS:
            controller.is_d_pressed = True
        elif action == glfw.RELEASE:
            controller.is_d_pressed = False

    if key == glfw.KEY_Q:
        if action == glfw.PRESS:
            controller.not_zone= not controller.not_zone

    if key == glfw.KEY_A:
        if action == glfw.PRESS:
            controller.zone = 1

    elif key == glfw.KEY_B:
        if action == glfw.PRESS:
            controller.zone = 2

    elif key == glfw.KEY_C:
        if action == glfw.PRESS:
            controller.zone = 3

#Funcion para inicializar el programa
def Setup():
    global controller
    file_name = sys.argv[1] # Se recibe como parametro el archivo json con las especificacion de los peces y la solucion de la edp
    #file_name = "view-setup.json"
    file = open(file_name) # Se abre y se guardan los datos
    data = json.load(file)
    solution = np.load(data["filename"]) #Se guarda la solucion del aquario
    dimension = np.array([solution.shape[0]-1, solution.shape[1]-1, solution.shape[2]-1]) # dimensiones del aquario

    #Se inicializa la escena
    scene.Setup(solution, data["t_a"], data["t_b"], data["t_c"], data["n_a"], data["n_b"], data["n_c"] )
    room_dim = scene.room_dim # dimensiones de la sala del aquario
    #Posicion donde aparece el usuario
    spawn_pos = np.array([
        dimension[0]/2 + room_dim[0]/2 + dimension[2]*(1-1/3),
        dimension[1]/2,
        dimension[2]/2
    ])
    #Se inicializa el controlador
    controller = user.Controller(dimension/2, spawn_pos)

#Funcion que actualiza el programa cada framee
def Update(delta):
    #proyeccion de la camara
    projection = tr.perspective(70, float(WIDTH) / float(HEIGHT), 0.1, 200)

    if (controller.fillPolygon):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    #Se obtiene la vista del usuario
    viewPos, view = user.CameraUpdate(controller, delta)
    #Se actualiza la escena
    scene.Update(delta, projection, view, viewPos, controller)

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    #Funcion para tomar 4 muestras por fragmento
    glfwWindowHint(GLFW_SAMPLES, 4)

    window = glfw.create_window(WIDTH, HEIGHT, "Aquarium View", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Setting up the clear screen color
    glClearColor(0.2, 0.2, 0.2, 1.0)

    #Se habilitan las transparencias
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)
    #Se habilita el muestreo multiple por pixel/fragmento
    glEnable(GL_MULTISAMPLE)
    #Se habilita el Stencil Test para crear las outlines
    glEnable(GL_STENCIL_TEST)
    #Se habilita el culling de caras, para dibujar solamente las caras necesarias
    glEnable(GL_CULL_FACE)

    #Inicializacion
    Setup()

    t0 = glfw.get_time()
    time_counter = 0

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        #Controlador de frames
        if time_counter > 1/FPS:
            #Actualizacion
            Update(time_counter)
            time_counter = 0
            # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
            glfw.swap_buffers(window)
        time_counter += dt
    glfw.terminate()