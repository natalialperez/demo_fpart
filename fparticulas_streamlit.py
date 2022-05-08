"""
Código de Filtro de Partículas para seguimiento en imágenes.

Natalia Pérez García de la Puente - 30.3.22

Usage: streamlit run fparticulas_streamlit.py
"""


import fparticulas as fp
import streamlit as st
import numpy as np
import cv2


def maskingcar(img):
    l_1 = np.array([0, 131, 0])
    u_1 = np.array([179, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, l_1, u_1)


def morphopercar(m):
    kernel = np.ones((21, 21), np.uint8)
    m = cv2.dilate(m, kernel, iterations=2)
    return m



if __name__ == '__main__':

    st.title('Filtro de partículas')
    st.sidebar.header('Selecciona un vídeo')
    videos_path = ["Media/bigball.mp4", "Media/twocars.mp4"]
    selected_video = st.sidebar.selectbox('Vídeo', videos_path)
    # _____________________________________________________
    st.sidebar.header('Parámetros de la función')
    npart = st.sidebar.slider('Nº partículas', 50, 100, 93)
    sizepart = st.sidebar.slider('Tamaño partículas', 25, 100, 95)
    noise = st.sidebar.slider('Diffusion noise', 20, 50, 28)
    agree = st.checkbox('Show step by step')
    # ______________________________________________________

    if selected_video is not None:

        vf = cv2.VideoCapture(selected_video)
        stframe = st.empty()
        placeholder = st.empty()
        noweighted_frame = True
        col1, col2, col3 = st.columns(3)

        while (vf.isOpened()):
            # Read a frame from video
            ret, frame = vf.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            resize_dim = 600
            max_dim = max(frame.shape)
            scale = resize_dim / max_dim
            image = cv2.resize(frame, None, fx=scale, fy=scale)
            imgsize = image.shape

            if videos_path == "Media/twocars.mp4":
                red_mask = maskingcar(image)
                mask = morphopercar(red_mask)
            else:
                red_mask = fp.masking(image)
                mask = fp.morphoper(red_mask)

            if noweighted_frame:
                particles = fp.init_particles(npart, sizepart, imgsize)  # Defined by [x1, y1, x2, y2]
                image_particles = fp.plot_particles(particles, image)  # Plot in blue
                weights = fp.evaluate(particles, mask)  # Calculate weights
                if any(t > 0.0 for t in weights):
                    noweighted_frame = False
                stframe.image(cv2.cvtColor(image_particles, cv2.COLOR_BGR2RGB))


            else:
                weights = fp.evaluate(particles, mask)  # Calculate weights
                estimated, coords = fp.estimation(particles, weights)  # Estimate max in weights
                particles2, weights2 = fp.selection(particles, weights)  # Defined by [x1, y1, x2, y2]
                particles = fp.diffusion(particles2, noise, sizepart)
                image_weights = fp.plot_estimated(image, estimated, coords)
                if all(t == 0.0 for t in weights):
                    noweighted_frame = True
                stframe.image(cv2.cvtColor(image_weights, cv2.COLOR_BGR2RGB))
                if agree:
                    with placeholder.container():
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.markdown('Inicialización partículas')
                        kpi1.image(cv2.cvtColor(image_particles, cv2.COLOR_BGR2RGB))
                        kpi2.markdown('Evaluación y estimación')
                        kpi2.image(fp.plot_weights(particles, weights, image, estimated))
                        kpi3.markdown('Selección y difusión')
                        kpi3.image(fp.plot_selection(particles2, weights2, image, estimated))

        vf.release()
        cv2.destroyAllWindows()

