# TFG: Detección de Personas y Objetos en Carretera

Este trabajo fin de grado se centra en ser capaz de proveer de conocimiento a la "máquina" para que sea capaz de detectar cualquier obstáculo en una situación de conducción, además de determinar si presentan un peligro o no para la carretera.

Los sistemas de videovigilancia automática para el control del tráfico rodado sirven a
múltiples finalidades. En este caso nos centraremos en la detección de los vehículos que
circulan por una vía rápida (autovía o autopista), realizando su seguimiento para
identificar su posición y velocidad. Además el sistema deberá estimar el tipo de vehículo
del que se trata: motocicleta, turismo, furgoneta o camión. El objetivo es usar toda esta
información para hacer una estimación del flujo de tráfico en la vía y la contaminación
generada. Para realizar esta detección se hará uso de redes neuronales
convolucionales. En particular emplearemos redes que son capaces de detectar
múltiples objetos en una escena. La programación se realizará en el lenguaje Python,
empleando bibliotecas de aprendizaje profundo.


## Comenzando 🚀

Para hacer una copia del proyecto en local:

1 - Dirigirte al directorio deseado y estar seguros de tener instalado git en el sistema.

2 - Ejecutar el siguiente comando:

```
git clone https://github.com/ramondfdez/tfg.git
```


### Pre-requisitos 📋

(Solo si se quiere ejecutar en local)

 * Git

```
$ sudo apt-get install git
```

 * Python 3 o superior (Python 2 ha dejado de tener soporte)

```
$ sudo apt install python3
```

 * Pip

```
$ sudo apt install python3-pip
```

* Instalar todas las librerías
```
$ pip3 install -r requirements.txt
```

### Ejecución 🔧

* En local:

  1 - Instalar las librerías y herramientas mencionadas en el punto anterior.
  
  2 - Hacer una copia en local del repositorio que se comentó en el primer punto.
  
  3 - Ejecutar el archivo mask_rcnn_video.py
  
     ```
     $  python mask_rcnn_video.py
     ```
  
  4 - Esperar a que acabe la ejecución del programa (depende si tu máquina está usando CPU o GPU para ello)
  
* En Google Colab

  1 - Ir a https://colab.research.google.com/drive/1DOAt9qMap963LsBzPkj5ygAkZr3WIu_0 donde se encuentra nuestro cuaderno.
  
  2 - Ejecutar cada uno de los bloques

## Construido con 🛠️

* [Python](https://www.python.org/) - Lenguaje de programación usado
* [OpenCV](https://github.com/opencv/opencv) - Librería de visión por computador
* [NumPy](https://github.com/numpy/numpy) - Librería de Data Science Fundamental
* [Imutils](https://github.com/jrosebr1/imutils) - Librería de manejo de imágenes básica para OpenCV 

## Versionado 📌

Actualmente esta es la versión 0.1

## Autores ✒️

* **Ramón Díaz Fernández** - *Trabajo - Documentación* - [ramondfdez](https://github.com/ramondfdez)

También puedes mirar la lista de todos los [contribuyentes](https://github.com/ramondfdez/tfg/blob/master/CONTRIBUTORS) que han participado en este proyecto. 

## Licencia 📄

Este proyecto está bajo la Licencia del MIT - mira el archivo [LICENSE.md](LICENSE.md) para más detalles

## Expresiones de Gratitud 🎁

* Agradecer a Ezequiel y a Miguel Ángel su tiempo y ayuda.
