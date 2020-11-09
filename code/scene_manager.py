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
import fishes as fh
import gpu_shapes as gpu

#VARIABLES GLOBALES

# shapes
gpu_voxels = []
gpu_outlines = []
gpu_objects = []
gpu_aquarium = []
gpu_led = None
Dim = 0 # dimensiones del aquario
room_dim = 0 # dimensiones de la sala

#Pipelines
Transparent_Pipeline = None
LightPipeline = 0
TexturePipeline = 0
ColorPipeline = 0
ColorTransparentPipeline = 0
NormalPipeline = 0

#Posicion de las fuentes de luz
Light0_pos = [0, 0, 0]
Light1_pos = [0, 0, 0]
Light2_pos = [0, 0, 0]
Light3_pos = [0, 0, 0]
#Parametros de iluminacion
Constant_att = 0
Linear_att = 0
Quadratic_att = 0.0
Shininess = 128

#Skybox
gpu_cube = 0
CubeMapPipeline = 0
skybox_texture = 0

#Funcion para la seleccion del color segun la temperatura
def ColorSelection(t_0, t_1, t_2, c_0, c_1, c_2):
    c = [c_0, c_1, c_2]
    t = [t_0, t_1, t_2]
    t.sort()
    return c[t.index(t_0)], c[t.index(t_1)], c[t.index(t_2)]

#Funcion para crear el cubeMap que sirve para los reflejos del aquario
def Setup_cube():
    global gpu_cube, CubeMapPipeline, skybox_texture
    CubeMapPipeline = fx.CubeMapShaderProgram()
    gpu_cube = es.toGPUShape(gpu.NormalCube(4), GL_REPEAT, GL_LINEAR)
    skybox_texture = fx.loadCubeMap([
        "sprites/cubemap/RIGHT.png",
        "sprites/cubemap/LEFT.png",
        "sprites/cubemap/FRONT.png",
        "sprites/cubemap/BACK.png",
        "sprites/cubemap/TOP.png",
        "sprites/cubemap/BOTTOM.png"
    ])
    gpu_cube.texture = skybox_texture

#Funcion para crear los elementos del acuario
def create_glases(dim, side_b, up_b, up_gap, offset):
    global gpu_cube, CubeMapPipeline, skybox_texture

    CubeMapPipeline = fx.CubeMapShaderProgram()

    W, D, H = dim[0], dim[1], dim[2]
    # CORNER POINTS
    sw_down = [0 - offset, 0 - offset, 0]
    se_down = [W + offset, 0 - offset, 0]
    nw_down = [0 - offset, D + offset, 0]
    ne_down = [W + offset, D + offset, 0]
    sw_up = [0 - offset, 0 - offset, H + up_gap]
    se_up = [W + offset, 0 - offset, H + up_gap]
    nw_up = [0 - offset, D + offset, H + up_gap]
    ne_up = [W + offset, D + offset, H + up_gap]

    #Vidrios transparentes
    up_c = [0.0, 0.2, 1.0]
    vertices_2 = [
        sw_down[0]+ side_b, sw_down[1], sw_down[2], 0, -1, 0,
        se_down[0]- side_b, se_down[1], se_down[2],  0, -1, 0,
        se_down[0]- side_b, se_down[1], se_up[2], 0, -1, 0,
        sw_down[0]+ side_b, sw_down[1], sw_up[2], 0, -1, 0,

        ne_down[0]- side_b, ne_down[1], ne_down[2],  0, 1, 0,
        nw_down[0]+ side_b, nw_down[1], nw_down[2],  0, 1, 0,
        nw_down[0]+ side_b, nw_down[1], nw_up[2],  0, 1, 0,
        ne_down[0]- side_b, ne_down[1], ne_up[2],  0, 1, 0,

        se_down[0], se_down[1]+ side_b, se_down[2],  1, 0, 0,
        ne_down[0], ne_down[1]- side_b, ne_down[2],  1, 0, 0,
        ne_down[0], ne_down[1]- side_b, ne_up[2],  1, 0, 0,
        se_down[0], se_down[1]+ side_b, se_up[2],  1, 0, 0,

        nw_down[0], nw_down[1]- side_b, nw_down[2],  -1, 0, 0,
        sw_down[0], sw_down[1]+ side_b, sw_down[2],  -1, 0, 0,
        sw_down[0], sw_down[1]+ side_b, sw_up[2],  -1, 0, 0,
        nw_down[0], nw_down[1]- side_b, nw_up[2],  -1, 0, 0,

        se_down[0], se_down[1], se_down[2] + H,  0, 0, 1,
        ne_down[0], ne_down[1], ne_down[2] + H,  0, 0, 1,
        nw_down[0], nw_down[1], nw_down[2] + H,  0, 0, 1,
        sw_down[0], sw_down[1], sw_down[2] + H,  0, 0, 1
    ]
    indices_2 = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16
    ]

    gpu_cube = es.toGPUShape(bs.Shape(vertices_2, indices_2, None), GL_REPEAT, GL_LINEAR)
    skybox_texture = fx.loadCubeMap([
        "sprites/cubemap/RIGHT.png",
        "sprites/cubemap/LEFT.png",
        "sprites/cubemap/FRONT.png",
        "sprites/cubemap/BACK.png",
        "sprites/cubemap/TOP.png",
        "sprites/cubemap/BOTTOM.png"
    ])
    gpu_cube.texture = skybox_texture

#Funcion para montar la escena
def Setup(solved_equation, t_a, t_b, t_c, n_a, n_b, n_c):
    global gpu_voxels, gpu_outlines,  gpu_objects, gpu_aquarium, gpu_led
    global Transparent_Pipeline, LightPipeline, TexturePipeline, ColorPipeline, ColorTransparentPipeline, NormalPipeline
    global room_dim, Dim, Light0_pos, Light1_pos, Light2_pos, Light3_pos, Constant_att, Linear_att, Quadratic_att


    #Se asignan los pipelines
    Transparent_Pipeline = fx.TransparentShaderProgram()
    LightPipeline = fx.FourTexturePhongShaderProgram()
    TexturePipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    ColorPipeline = fx.FourPhongShaderProgram()
    ColorTransparentPipeline = fx.FourPhongTransparentShaderProgram()
    NormalPipeline = es.SimpleModelViewProjectionShaderProgram()

    #Se establece las dimensiones del aquario
    Dim = solved_equation.shape
    print(Dim)

    c_a, c_b, c_c = ColorSelection(t_a, t_b, t_c,
                                   c_0 = [0.8, 0, 1],
                                   c_1 = [1, 0, 0],
                                   c_2 = [1, 0.7, 0])
    #Se crean los voxeles para la t_a a dibujar, las posiciones donde se pueden generar peces y voxeles en los que se pueden mover
    shape_a, a_positions, a_voxels = gpu.create_solid_zone(solved_equation, t_a, c_a)
    gpu_voxels += [es.toGPUShape(shape_a)]
    #Se crea la shape para la outline, version escalada de los voxeles anteriores
    gpu_outlines += [es.toGPUShape(gpu.create_outline_zone(solved_equation, t_a, c_a, 0.4))]

    # Se crean los voxeles para la t_b a dibujar, las posiciones donde se pueden generar peces y voxeles en los que se pueden mover
    shape_b, b_positions, b_voxels = gpu.create_solid_zone(solved_equation, t_b, c_b)
    gpu_voxels += [es.toGPUShape(shape_b)]
    # Se crea la shape para la outline, version escalada de los voxeles anteriores
    gpu_outlines += [es.toGPUShape(gpu.create_outline_zone(solved_equation, t_b, c_b, 0.4))]

    # Se crean los voxeles para la t_c a dibujar, las posiciones donde se pueden generar peces y voxeles en los que se pueden mover
    shape_c, c_positions, c_voxels = gpu.create_solid_zone(solved_equation, t_c, c_c)
    gpu_voxels += [es.toGPUShape(shape_c)]
    # Se crea la shape para la outline, version escalada de los voxeles anteriores
    gpu_outlines += [es.toGPUShape(gpu.create_outline_zone(solved_equation, t_c, c_c, 0.4))]

    #Se generan los peces
    fh.Setup(a_positions, b_positions, c_positions, n_a, n_b, n_c, Dim, a_voxels, b_voxels, c_voxels)

    #Se establecen las dimensiones de la sala
    max_side = max(Dim[0], Dim[1])
    room_side = max_side*2
    lower_aquarium = Dim[2]/2
    upper_aquarium = Dim[2]
    room_dim = [max_side+room_side, max_side+room_side, Dim[2] + upper_aquarium + lower_aquarium]
    #Se genera la sala
    gpu_objects = gpu.create_room(Dim, room_side/2, lower_aquarium, upper_aquarium)
    #Se genera el aquario
    gpu_aquarium = gpu.create_aquarium(Dim, 0.3, Dim[2]/10, Dim[2]/20, 0.1)
    #Se generan los vidrios del acuario
    create_glases(Dim, 0.3, Dim[2]/10, Dim[2]/20, 0.1)
    #Se generan los leds
    gpu_led = es.toGPUShape(gpu.create_led([0.7, 0.7, 0.7],[0.4,0.4,0.4], 32))

    #Se establece las pociones de las fuentes de luz
    light_height = Dim[2] + upper_aquarium - Dim[2]/5
    ld = 0.35
    Light0_pos = [Dim[0] / 2 - room_side * ld, Dim[1] / 2 - room_side * ld, light_height]
    Light1_pos = [Dim[0] / 2 + room_side * ld, Dim[1] / 2 - room_side * ld, light_height]
    Light2_pos = [Dim[0] / 2 + room_side * ld, Dim[1] / 2 + room_side * ld, light_height]
    Light3_pos = [Dim[0] / 2 - room_side * ld, Dim[1] / 2 + room_side * ld, light_height]

    #Se regulan los parametros de iluminacion segun la dimension de la sala
    ligh_radius = room_dim[2]
    Constant_att = 0.25
    Linear_att = 2/ligh_radius
    Quadratic_att = 3/(ligh_radius**2)
#Funcion para preparar un shaderProgram antes de dibujar un objeto
def UsePipeline(shaderProgram, light, proj, view, v_pos=None):
    glUseProgram(shaderProgram)

    #Solo si tiene iluminacion, ajusta sus parametros
    if light:
        glUniform3f(glGetUniformLocation(shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos0"), Light0_pos[0], Light0_pos[1],
                    Light0_pos[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos1"), Light1_pos[0], Light1_pos[1],
                    Light1_pos[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos2"), Light2_pos[0], Light2_pos[1],
                    Light2_pos[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos3"), Light3_pos[0], Light3_pos[1],
                    Light3_pos[2])
        glUniform3f(glGetUniformLocation(shaderProgram, "viewPosition"), v_pos[0], v_pos[1],
                    v_pos[2])
        glUniform1ui(glGetUniformLocation(shaderProgram, "shininess"), Shininess)

        glUniform1f(glGetUniformLocation(shaderProgram, "constantAttenuation"), Constant_att)
        glUniform1f(glGetUniformLocation(shaderProgram, "linearAttenuation"), Linear_att)
        glUniform1f(glGetUniformLocation(shaderProgram, "quadraticAttenuation"), Quadratic_att)

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_TRUE, proj)
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_TRUE, view)

#Funcion para dibujar un shape con o sin material
def Draw_Shapes(pipeline, shapes, transform, useMaterial, k_a =None, k_d=None, k_s=None):
    if useMaterial:
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), k_a[0], k_a[1], k_a[2])
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), k_d[0], k_d[1], k_d[2])
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), k_s[0], k_s[1], k_s[2])
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, transform)
    for i in range(len(shapes)):
        pipeline.drawShape(shapes[i])

#Funcion para dibujar los elementos de la sala
def Draw_room(proj, view, v_pos):
    #AQUARIUM BASE
    Draw_Shapes(LightPipeline, [gpu_objects[0]], tr.identity(), useMaterial=True,
               k_a = [0.5, 0.5, 0.5],
               k_d = [0.7, 0.7, 0.7],
               k_s = [1.2, 1.2, 1.2])

    #AQUARIUM GRAVEL
    Draw_Shapes(LightPipeline, [gpu_objects[1]], tr.identity(), useMaterial=True,
               k_a=[0.4, 0.4, 0.4],
               k_d=[0.5, 0.5, 0.5],
               k_s=[1.2, 1.2, 1.2])

    #ROOM FLOOR
    Draw_Shapes(LightPipeline, [gpu_objects[2]], tr.identity(), useMaterial=True,
               k_a=[0.4, 0.4, 0.4],
               k_d=[0.5, 0.5, 0.5],
               k_s=[1.7, 1.7, 1.7])

    #ROOM WALLS
    Draw_Shapes(LightPipeline, [gpu_objects[3], gpu_objects[4]], tr.identity(), useMaterial=True,
                k_a=[0.5, 0.5, 0.5],
                k_d=[0.6, 0.6, 0.6],
                k_s=[0.8, 0.8, 0.8])

    #ROOM CEILING
    Draw_Shapes(LightPipeline, [gpu_objects[5]], tr.identity(), useMaterial=True,
                k_a=[0.4, 0.4, 0.4],
                k_d=[0.5, 0.5, 0.5],
                k_s=[0.7, 0.7, 0.7])

    #POSTERS
    Draw_Shapes(LightPipeline, [gpu_objects[9], gpu_objects[10], gpu_objects[11]], tr.identity(), useMaterial=True,
                k_a=[0.4, 0.4, 0.4],
                k_d=[0.7, 0.7, 0.7],
                k_s=[1.0, 1.0, 1.0])

    #ENTRANCE
    UsePipeline(ColorPipeline.shaderProgram, light=True, proj=proj, view=view, v_pos=v_pos)
    Draw_Shapes(ColorPipeline, [gpu_objects[6]], tr.identity(), useMaterial=True,
                k_a=[0.7, 0.7, 0.7],
                k_d=[1.0, 1.0, 1.0],
                k_s=[1.9, 1.9, 1.9])

    #REGULADORES TERMICOS
    Draw_Shapes(ColorPipeline, [gpu_objects[7], gpu_objects[8]], tr.identity(), useMaterial=True,
                k_a=[0.4, 0.4, 0.4],
                k_d=[0.8, 0.8, 0.8],
                k_s=[2.5, 2.5, 2.5])


    #LEDS
    Draw_Shapes(ColorPipeline, [gpu_led],
                transform = tr.matmul([
                    tr.translate(Light0_pos[0], Light0_pos[1], Light0_pos[2] + Dim[2] / 5),
                    tr.scale(4,4,3)
                ]),
                useMaterial=True,
                k_a=[0.3, 0.3, 0.3],
                k_d=[0.6, 0.6, 0.6],
                k_s=[0.8, 0.8, 0.8])

    Draw_Shapes(ColorPipeline, [gpu_led],
                transform=tr.matmul([
                    tr.translate(Light1_pos[0], Light1_pos[1], Light1_pos[2] + Dim[2] / 5),
                    tr.scale(4,4,3)
                ]),
                useMaterial=True,
                k_a=[0.3, 0.3, 0.3],
                k_d=[0.6, 0.6, 0.6],
                k_s=[0.8, 0.8, 0.8])

    Draw_Shapes(ColorPipeline, [gpu_led],
                transform=tr.matmul([
                    tr.translate(Light2_pos[0], Light2_pos[1], Light2_pos[2] + Dim[2] / 5),
                    tr.scale(4,4,3)
                ]),
                useMaterial=True,
                k_a=[0.3, 0.3, 0.3],
                k_d=[0.6, 0.6, 0.6],
                k_s=[0.8, 0.8, 0.8])

    Draw_Shapes(ColorPipeline, [gpu_led],
                transform=tr.matmul([
                    tr.translate(Light3_pos[0], Light3_pos[1], Light3_pos[2] + Dim[2] / 5),
                    tr.scale(4,4,3)
                ]),
                useMaterial=True,
                k_a=[0.3, 0.3, 0.3],
                k_d=[0.6, 0.6, 0.6],
                k_s=[0.8, 0.8, 0.8])

#Funcion para dibujar el aquario
def Draw_aquarium(proj, view, v_pos):

    glDisable(GL_CULL_FACE)
    #Se desabilita en esta parte para ver por ambas caras de estas figuras
    UsePipeline(ColorPipeline.shaderProgram, light=True, proj=proj, view=view, v_pos=v_pos)
    Draw_Shapes(ColorPipeline, [gpu_aquarium[0], gpu_aquarium[1]], tr.identity(), useMaterial=True,
                k_a=[0.5, 0.5, 0.5],
                k_d=[0.6, 0.6, 0.6],
                k_s=[2.0, 2.0, 2.0])
    glEnable(GL_CULL_FACE)

    Draw_Shapes(ColorPipeline, [gpu_aquarium[2]], tr.identity(), useMaterial=True,
                k_a=[0.5, 0.5, 0.5],
                k_d=[0.6, 0.6, 0.6],
                k_s=[2.0, 2.0, 2.0])

    #Dibujamos el vidrio del aquario con lor reflejos
    UsePipeline(CubeMapPipeline.shaderProgram, light=False, proj=proj, view=view, v_pos=v_pos)
    glUniform1i(glGetUniformLocation(CubeMapPipeline.shaderProgram, "skybox"), 0)
    glUniform3f(glGetUniformLocation(CubeMapPipeline.shaderProgram, "cameraPos"), v_pos[0], v_pos[1],
                v_pos[2])
    glBindVertexArray(gpu_cube.vao)
    glBindTexture(GL_TEXTURE_CUBE_MAP, skybox_texture)
    Draw_Shapes(CubeMapPipeline, [gpu_cube], tr.translate(0, 0, 0), useMaterial=False)
    glBindVertexArray(0)


#Funcion que actualiza lña escena
def Update(delta, proj, view, v_pos,  ctrl):
    #Especificamos la accion a realizar despues del stencil test
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)

    # Clearing the screen in both, color and depth
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

    #Actualizamos los peces
    UsePipeline(LightPipeline.shaderProgram, light=True, proj=proj, view=view, v_pos=v_pos)
    fh.Update(delta)

    #Nos aseguramos de no escribir sobre el stencil buffer al dibujar la sala
    glStencilMask(0x00)
    #Dibujamos la sala
    Draw_room(proj, view, v_pos)

    # Stencil / Outline
    #Escribimos 1s en el stencil buffer en los fragmentos donde se renderiza los voxeles
    glStencilFunc(GL_ALWAYS, 1, 0xFF)
    glStencilMask(0xFF) #Habilitamos escribir sobre el stencil buffer

    #Dibujamos los voxeles actuales con transparencia
    if not ctrl.not_zone:
        UsePipeline(Transparent_Pipeline.shaderProgram, light=False, proj=proj, view=view)
        Draw_Shapes(Transparent_Pipeline, [gpu_voxels[ctrl.zone - 1]], tr.identity(), useMaterial=False)

    #Deshabiltamos escribir sobre el stencil buffer
    glStencilMask(0x00)
    #Se dibuja el aquario
    Draw_aquarium(proj, view, v_pos)

    if not ctrl.not_zone:
        #Solo se dibuja en las partes en que el stencil buffer tenga un valor distinto de 1
        glStencilFunc(GL_NOTEQUAL, 1, 0xFF)
        #Desactivamos el depth test para que se ve completo el outline
        glDisable(GL_DEPTH_TEST)

        #Dibujamos el outline
        UsePipeline(NormalPipeline.shaderProgram, light=False, proj=proj, view=view)
        Draw_Shapes(NormalPipeline, [gpu_outlines[ctrl.zone - 1]], tr.identity(), useMaterial=False)

    #Volvemos a los paremtros normales
    glStencilMask(0xFF)
    glStencilFunc(GL_ALWAYS, 1, 0xFF)
    glEnable(GL_DEPTH_TEST)




