from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from PIL import Image

import basic_shapes as bs
import easy_shaders as es


# We will use 32 bits data, so we have 4 bytes
# 1 byte = 8 bits
SIZE_IN_BYTES = 4

#Funcion para cargar un cubeMap
def loadCubeMap(faces_files):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
    for i in range(len(faces_files)):
        image = Image.open(faces_files[i])
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

        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internalFormat, image.size[0], image.size[1], 0, format, GL_UNSIGNED_BYTE,
                     img_data)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return texture

#Pipeline para los reflejos del acuario
class CubeMapShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 normal;
            
            out vec3 Position;
            out vec3 Normal;
            
            void main()
            {   Normal = mat3(transpose(inverse(model))) * normal;
                Position = vec3(model * vec4(position, 1.0));
                gl_Position = projection * view *  vec4(Position, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core
            in vec3 Normal;
            in vec3 Position;

            out vec4 FragColor;
            
            uniform vec3 cameraPos;
            uniform samplerCube skybox;
            
            void main()
            {   
                vec3 I = normalize(Position - cameraPos);
                vec3 R = reflect(I, normalize(Normal));
                FragColor = vec4(texture(skybox, R).rgb + vec3(0, 0, 0.3), 0.4);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color specification => 3*4 + 3*4 = 24 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(normal)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)

#Pipeline simple transparente
class TransparentShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 130

            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;

            in vec3 position;
            in vec3 color;

            out vec3 newColor;
            void main()
            {
                gl_Position = projection * view * model * vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 130
            in vec3 newColor;

            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 0.4f);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color specification => 3*4 + 3*4 = 24 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)

#Pipeline con 4 focos de luz, transparente
class FourPhongTransparentShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec3 normal;

            out vec3 fragPosition;
            out vec3 fragOriginalColor;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragOriginalColor = color;
                fragNormal = mat3(transpose(inverse(model))) * normal;  

                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            out vec4 fragColor;

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec3 fragOriginalColor;

            uniform vec3 lightPos0; 
            uniform vec3 lightPos1; 
            uniform vec3 lightPos2; 
            uniform vec3 lightPos3; 
            uniform vec3 viewPosition;
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;
            uniform float alpha;

            void main()
            {
                // ambient
                vec3 ambient = Ka * La;

                // diffuse
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0, 0, 0);
                vec3 lights[4] = vec3[](lightPos0, lightPos1, lightPos2, lightPos3);

                for(int i = 0; i < 4; i++)
                {
                    vec3 toLight = lights[i] - fragPosition;
                    vec3 lightDir = normalize(toLight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Ld * diff;

                    // specular
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    vec3 halfwayDir = normalize(lightDir + viewDir);
                    float spec = pow(max(dot(normalizedNormal, halfwayDir), 0.0), shininess);
                    vec3 specular = Ks * Ls * spec;

                    // attenuation
                    float distToLight = length(toLight);
                    float attenuation = constantAttenuation
                        + linearAttenuation * distToLight
                        + quadraticAttenuation * distToLight * distToLight;

                    result += (ambient + diffuse + specular) / attenuation;
                }
                result *= fragOriginalColor;
                fragColor = vec4(result, alpha);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        glEnableVertexAttribArray(normal)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)

#Pipeline con 4 focos de luz
class FourPhongShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec3 normal;

            out vec3 fragPosition;
            out vec3 fragOriginalColor;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragOriginalColor = color;
                fragNormal = mat3(transpose(inverse(model))) * normal;  

                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            out vec4 fragColor;

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec3 fragOriginalColor;

            uniform vec3 lightPos0; 
            uniform vec3 lightPos1; 
            uniform vec3 lightPos2; 
            uniform vec3 lightPos3; 
            uniform vec3 viewPosition;
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            void main()
            {
                // ambient
                vec3 ambient = Ka * La;

                // diffuse
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0, 0, 0);
                vec3 lights[4] = vec3[](lightPos0, lightPos1, lightPos2, lightPos3);
                
                for(int i = 0; i < 4; i++)
                {
                    vec3 toLight = lights[i] - fragPosition;
                    vec3 lightDir = normalize(toLight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Ld * diff;
    
                    // specular
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    vec3 halfwayDir = normalize(lightDir + viewDir);
                    float spec = pow(max(dot(normalizedNormal, halfwayDir), 0.0), shininess);
                    vec3 specular = Ks * Ls * spec;
    
                    // attenuation
                    float distToLight = length(toLight);
                    float attenuation = constantAttenuation
                        + linearAttenuation * distToLight
                        + quadraticAttenuation * distToLight * distToLight;
                    
                    result += (ambient + diffuse + specular) / attenuation;
                }
                result *= fragOriginalColor;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        glEnableVertexAttribArray(normal)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)

#Pipeline con 4 focos de luz, con texturas
class FourTexturePhongShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            in vec3 position;
            in vec2 texCoords;
            in vec3 normal;

            out vec3 fragPosition;
            out vec2 fragTexCoords;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragTexCoords = texCoords;
                fragNormal = mat3(transpose(inverse(model))) * normal;  

                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec2 fragTexCoords;

            out vec4 fragColor;

            uniform vec3 lightPos0; 
            uniform vec3 lightPos1; 
            uniform vec3 lightPos2; 
            uniform vec3 lightPos3; 
            uniform vec3 viewPosition; 
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            uniform sampler2D samplerTex;

            void main()
            {
                // ambient
                vec3 ambient = Ka * La;

                // diffuse
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0, 0, 0);
                vec3 lights[4] = vec3[](lightPos0, lightPos1, lightPos2, lightPos3);
                
                for(int i = 0; i < 4; i++)
                {
                    vec3 toLight = lights[i] - fragPosition;
                    vec3 lightDir = normalize(toLight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Ld * diff;
    
                    // specular
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    vec3 halfwayDir = normalize(lightDir + viewDir);
                    float spec = pow(max(dot(normalizedNormal, halfwayDir), 0.0), shininess);
                    vec3 specular = Ks * Ls * spec;
    
                    // attenuation
                    float distToLight = length(toLight);
                    float attenuation = constantAttenuation
                        + linearAttenuation * distToLight
                        + quadraticAttenuation * distToLight * distToLight;
                    result += (ambient + diffuse + specular) / attenuation;
                }
                
                vec4 fragOriginalColor = texture(samplerTex, fragTexCoords);
                    
                result *= fragOriginalColor.rgb;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)
        glBindTexture(GL_TEXTURE_2D, shape.texture)

        # 3d vertices + 2d texture coordinates + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        texCoords = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(texCoords)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        glEnableVertexAttribArray(normal)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)