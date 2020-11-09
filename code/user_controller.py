from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

import transformations as tr

import scene_manager as scene

#Camara Esferica
class SphericCamera:
    def __init__(self):
        self.center = np.array([0, 0, 0])

        self.phi_angle = np.pi
        self.theta_angle = - np.pi / 2
        self.eyeX = 0.0
        self.eyeY = 0.0
        self.eyeZ = 0.0
        self.viewPos = np.zeros(3)
        self.view = 0.0
        self.radius = 20
        self.up = np.array([0, 0, 1])

    def change_theta_angle(self, dt):
        self.theta_angle = (self.theta_angle + dt) % (np.pi * 2)

    def change_phi_angle(self, dt):
        self.phi_angle = (self.phi_angle + dt) % (np.pi * 2)

    def change_zoom(self, dr):
        if self.radius + dr > 0.1:
            self.radius += dr

    def update_view(self):
        self.eyeX = self.radius * np.sin(self.theta_angle) * np.cos(self.phi_angle) + self.center[0]
        self.eyeY = self.radius * np.sin(self.theta_angle) * np.sin(self.phi_angle) + self.center[1]
        self.eyeZ = self.radius * np.cos(self.theta_angle) + self.center[2]

        up_x = np.cos(self.theta_angle) * np.cos(self.phi_angle) * np.array([1, 0, 0])
        up_y = np.cos(self.theta_angle) * np.sin(self.phi_angle) * np.array([0, 1, 0])
        up_z = - np.sin(self.theta_angle) * np.array([0, 0, 1])
        self.up = up_x + up_y + up_z

        self.viewPos = np.array([self.eyeX, self.eyeY, self.eyeZ])
        self.view = tr.lookAt(
            self.viewPos,
            self.center,
            self.up
        )
        return self.view

# A class to store the application control
class Controller:
    def __init__(self, pos, spawn):
        self.fillPolygon = True
        self.showAxis = True

        self.is_up_pressed = False
        self.is_down_pressed = False
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_space_press = False
        self.is_z_pressed = False
        self.is_x_pressed = False
        self.is_e_pressed = False
        self.is_d_pressed = False
        self.is_s_pressed = False
        self.is_f_pressed = False
        self.not_zone = True #Indica si se deben mostrar los voxeles
        self.is_ctrl_pressed = False #indica si se esta agachado
        self.fp_view = True # Indica si la camara esta en primera persona
        self.zone = 1 # zona actual a dibujar

        self.cam_pos = spawn
        self.cam_phi = np.pi
        self.cam_theta = np.pi/2

        self.spheric_camera = SphericCamera()
        self.spheric_camera.center = pos

        self.velocity = max(scene.Dim[0], scene.Dim[1])*2/3 # velocidad de movimiento en la escena

    @property
    def camera(self):
        """ Get a camera reference from the controller object. """
        return self.spheric_camera

#Funcion para manejar la camara dado el input
def InputToCamera(control, camera, delta):
    # La camara esta en primera persona
    if control.fp_view:
        #Manejo de las rotaciones, limitando el angulo theta(rotar para  arriba o abajo)
        if control.is_left_pressed:
            control.cam_phi = (control.cam_phi + delta*1.5)%(2*np.pi)
        if control.is_right_pressed:
            control.cam_phi = (control.cam_phi - delta*1.5)%(2*np.pi)
        if control.is_up_pressed and control.cam_theta > np.pi/4:
            control.cam_theta = (control.cam_theta - delta*1.5)%(2*np.pi)
        if control.is_down_pressed and control.cam_theta < np.pi - np.pi/8:
            control.cam_theta = (control.cam_theta + delta*1.5)%(2*np.pi)

        #Vector apuntando en la direccion del movimiento del usuario
        forward = np.array([np.cos(control.cam_phi), np.sin(control.cam_phi), 0])
        #Vector apuntando en la direccion izquierda
        side = np.array([np.cos(control.cam_phi + np.pi/2), np.sin(control.cam_phi + np.pi/2), 0])
        #El input genera una posicion tentativa
        tent_pos = control.cam_pos
        if control.is_e_pressed:
            tent_pos = tent_pos + forward * delta * control.velocity
        if control.is_d_pressed:
            tent_pos = tent_pos - forward * delta * control.velocity
        if control.is_s_pressed:
            tent_pos = tent_pos + side * delta * control.velocity
        if control.is_f_pressed:
            tent_pos = tent_pos - side * delta * control.velocity

        # CControla si el usuario esta agachado o no
        if control.is_ctrl_pressed:
            control.cam_pos[2] = scene.Dim[2]/8
        else:
            control.cam_pos[2] = scene.Dim[2]/2

        room = scene.room_dim
        dim = scene.Dim
        #Se manejan las colisiones, Si la posicion tentativa no se encuentra fuera del dominio en el que se puede mover
        #(fuera de la sala o dentro del acuario) el usuario se mueve a la posicion tentativa
        if (tent_pos[1] > (dim[1] / 2 - room[1] / 2) and tent_pos[1] < 0 or
                tent_pos[1] < (dim[1] / 2 + room[1] / 2) and tent_pos[1] > dim[1]):
            if (tent_pos[0] > (dim[0]/2 - room[0]/2) and tent_pos[0] < (dim[0]/2 + room[0]/2)):
                control.cam_pos = tent_pos
        elif tent_pos[1] >= 0 and tent_pos[1] <= dim[1]:
            if (tent_pos[0] > (dim[0] / 2 - room[0] / 2) and tent_pos[0] < 0 or
                    tent_pos[0] < (dim[0] / 2 + room[0] / 2 + dim[2]) and tent_pos[0] > dim[0]):
                control.cam_pos = tent_pos
    #Se mueve camara esferica
    else:
        if control.is_left_pressed:
            camera.change_phi_angle(-1 * delta)

        if control.is_right_pressed:
            camera.change_phi_angle( 1 * delta)

        if control.is_up_pressed:
            camera.change_theta_angle( 1 * delta)

        if control.is_down_pressed:
            camera.change_theta_angle(-1 * delta)

        if control.is_x_pressed:
            camera.change_zoom(10 * delta)

        if control.is_z_pressed:
            camera.change_zoom(-10 * delta)

#Funcion para actualizar la camara segun su tipo de visualizacion actual
def CameraUpdate(controller, delta):
    camera = controller.camera

    InputToCamera(controller, camera, delta)

    if controller.fp_view: #Camara se encuentra en primera persona
        viewPos = controller.cam_pos
        rad = 2
        look_at =  viewPos + np.array([
            rad * np.sin(controller.cam_theta)*np.cos(controller.cam_phi),
            rad * np.sin(controller.cam_theta) * np.sin(controller.cam_phi),
            rad * np.cos(controller.cam_theta)
        ])
        up = np.array([0,0,1])
        view = tr.lookAt(
            viewPos,
            look_at,
            up
        )
    else: #Camara esferica
        view = camera.update_view()
        viewPos = camera.viewPos

    return viewPos, view
