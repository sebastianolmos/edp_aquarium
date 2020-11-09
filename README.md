# EDP Aquarium
Tercera tarea del curso Modelación y Computación Grafica, otoño 2020, usando OpenGL para hacer una aplicación 3D que represente un acuario que contiene tres 
tipos de peces, donde cada tipo se encuentra en su zona de temperatura preferida, para esto se resuelve una ecuacion diferencial parcial en tres dimensiones, simulando la transferencia de calor en el acuario.

Para ejecutar el programa debe tener instalado Python 3.5 o superior, y las librerias numpy, pyopengl, glfw, pillow y scipy.
Primero se tiene que resolver la EDP ejecutando en el directorio code :``python aquarium-solver.py problem-setup.json``, donde el ultimo argumento corresponde al archivo que contiene los parametros del acuario y valores de temperaturas. Luego se ejecuta ``python aquarium-view.py view-setup.json`` para iniciar la visualizacion, donde el ultimo argumento es el archivo que contiene los paranetros de los tipos de peces. Se recomienda usar los valores de los archivo ya dados, por lo que solo necesita correr el ultimo comando para apreciar el programa.

Video de muestra:

[![Alt text for your video](https://img.youtube.com/vi/6nktvC8u2hk/0.jpg)](https://youtu.be/6nktvC8u2hk)

Los controles generales son:
- [Q]  Activar la visualizacion de los voxeles de T° preferida por tipo de pez
- [A], [B], [C] para cambiar la zona de T° a la del otro tipo de pez
- [V] cambiar la cámara

Los controles para la camara global son:
- Las Flechas de dirección sirven para rotar la cámara.
- [Z], [X] para acercarse y alejarse al centro de la pista.

Los controles para la camara en primera persona son:
- Las Flechas de dirección sirven para rotar la cámara en torno a la pista
- [E], [D], [S], [F] para moverse.
- [Ctrl izq] para agacharse/pararse.
