"""
Código de Filtro de Partículas para seguimiento en imágenes.

Natalia Pérez García de la Puente - 30.3.22

Usage: python fparticulas.py -i SecuenciaPelota
"""


import os
import argparse
import numpy as np
from numpy.random import uniform
import cv2
from natsort import natsorted
from itertools import accumulate


def plot_estimated(image, estimated, coords):
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    thickness = 2
    [x1, y1, x2, y2] = coords
    start_point = (x1, y1)
    end_point = (x2, y2)
    color = (0, 0, 255)
    cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.putText(image, str(round(estimated, 2)), (x1, y1 - 5), font, font_scale,
                color, thickness, cv2.LINE_AA, False)
    return image


def gaussian_noise(x, sigma):
    # Random samples from a normal (Gaussian) distribution.
    mu = 0  # mean and standard deviation
    size = 1
    s = np.random.normal(mu, sigma, size)
    return x + s


def t_noise(x, sigma):
    # Random samples from a standard Student’s t distribution with df degrees of freedom.
    df = 1.0  # Degrees of freedom, must be > 0.
    size = 1
    n = np.random.standard_t(df, size) * sigma
    return x + n


def cauchy_noise(x, sigma):
    # Random samples from a standard Cauchy distribution with mode = 0.
    size = 1
    n = np.random.standard_cauchy(size) * sigma
    return x + n


def diffusion(particles, sigma, sizepart):
    # A las partículas seleccionadas en el instante anterior, se les aplicará una perturbación
    # aleatoria con una distribución a vuestra elección. Habitualmente, esta distribución es gausiana.
    new_particles = []
    for particle in particles:
        [x1, y1, _, _] = particle
        noise_x = int(gaussian_noise(x1, sigma))
        noise_y = int(gaussian_noise(y1, sigma))
        x_noise2 = noise_x + sizepart
        y_noise2 = noise_y + sizepart
        new_particles.append([noise_x, noise_y, x_noise2, y_noise2])
    return new_particles


def plot_selection(particles, weights, image, estimated):
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    thickness = 2
    for particle, weight in zip(particles, weights):
        [x1, y1, x2, y2] = particle
        start_point = (x1, y1)
        end_point = (x2, y2)
        color = (0, 0, 255) if weight == estimated else (180, 0, 180)
        cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(image, str(round(weight, 2)), (x1, y1 - 5), font, font_scale,
                    color, thickness, cv2.LINE_AA, False)
    return image


def selection(particles, weights):
    # El nuevo conjunto de partículas se genera remuestreando con reemplazo veces sobre la población actual.
    # La probabilidad de elegir una partícula está relacionada con el valor de su peso
    # La manera clásica de hacerlo es mediante el método de la ruleta
    accum_weights = list(accumulate(weights))
    new_particles = []
    new_weights = []
    for _ in range(len(particles)):
        ran_float = np.random.rand()
        particle_index = accum_weights.index(min([i for i in accum_weights if i > ran_float]))
        new_particles.append(particles[particle_index])
        new_weights.append(weights[particle_index])
    return new_particles, new_weights


def plot_weights(particles, weights, image, estimated):
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    thickness = 2
    for particle, weight in zip(particles, weights):
        [x1, y1, x2, y2] = particle
        start_point = (x1, y1)
        end_point = (x2, y2)
        color = (0, 0, 255) if weight == estimated else (0, 255, 0)
        cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(image, str(round(weight, 2)), (x1, y1 - 5), font, font_scale,
                    color, thickness, cv2.LINE_AA, False)
    return image


def estimation(particles, weights):
    # Selección de la partícula con mayor peso.
    max_value = np.max(weights)
    index_value = np.where(weights == max_value)
    index_array = np.asarray(index_value).astype('int32')
    part_value = particles[int(index_array[0][0])]
    return max_value, part_value


def evaluate(particles, mask):
    # Cálculo del peso de cada partícula utilizando una función de verosimilitud.
    weights = np.zeros(len(particles))
    total_sum = np.sum(mask)
    for i, particle in enumerate(particles):
        [x1, y1, x2, y2] = particle
        patch = mask[y1:y2, x1:x2]
        weights[i] = np.sum(patch) / (total_sum + 1e-12)
    weights = weights / (np.sum(weights) + 1e-15)
    return weights


def plot_particles(particles, mask):
    color = (255, 0, 0)
    thickness = 2
    for particle in particles:
        [x1, y1, x2, y2] = particle
        start_point = (x1, y1)
        end_point = (x2, y2)
        cv2.rectangle(mask, start_point, end_point, color, thickness)
    return mask


def init_particles(npart, sizepart, imgsize):
    # Esta etapa lleva a cabo una inicialización del estado de las partículas de tipo aleatorio sobre toda la imagen.
    particles = []
    for _ in range(npart):
        y1 = np.random.randint(imgsize[0] - sizepart)
        x1 = np.random.randint(imgsize[1] - sizepart)
        x2 = x1 + sizepart
        y2 = y1 + sizepart
        particles.append([x1, y1, x2, y2])
    return particles


def morphoper(m):
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.erode(m, kernel, iterations=2)
    m = cv2.dilate(m, kernel, iterations=2)
    return m


def masking(img):
    l_1 = np.array([0, 50, 50])
    u_1 = np.array([10, 255, 255])
    l_2 = np.array([170, 50, 50])
    u_2 = np.array([180, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, l_1, u_1)
    m2 = cv2.inRange(hsv, l_2, u_2)
    return m1 + m2


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to where the images reside")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = parse_arguments()
    path_in = args['images']
    npart = 50  # Entre 50-100 es buen número
    sizepart = 30
    noise = 25

    cont = natsorted(os.listdir(path_in))
    noweighted_frame = True

    for frame in cont:
        if os.path.isfile(os.path.join(path_in, frame)) and frame.endswith('.jpg'):
            image = cv2.imread(os.path.join(path_in, frame))
            imgsize = image.shape
            red_mask = masking(image)
            mask = morphoper(red_mask)

            if noweighted_frame:
                particles = init_particles(npart, sizepart, imgsize)  # Defined by [x1, y1, x2, y2]
                image_particles = plot_particles(particles, image)  # Plot in blue
                weights = evaluate(particles, mask)  # Calculate weights
                if any(t > 0.0 for t in weights):
                    noweighted_frame = False
                cv2.imshow(str(frame), image_particles)

            else:
                weights = evaluate(particles, mask)  # Calculate weights
                estimated, coords = estimation(particles, weights)  # Estimate max in weights
                # image_weights = plot_weights(particles, weights, image, estimated)  # Plot nearest in red
                particles2, weights2 = selection(particles, weights)  # Defined by [x1, y1, x2, y2]
                particles = diffusion(particles2, noise, sizepart)
                # image_weights = plot_selection(particles2, weights2, image, estimated)  # Plot in pink
                image_weights = plot_estimated(image, estimated, coords)
                if all(t == 0.0 for t in weights):
                    noweighted_frame = True

                cv2.imshow(str(frame), image_weights)

            cv2.waitKey(1)

    cv2.destroyAllWindows()
