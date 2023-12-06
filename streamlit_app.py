import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Crear el título de la aplicación y un widget oara carcar un archivo.
st.title("Detección de rostros basados en aprendizaje profundo")
img_file_buffer = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'])


# Función para detectar rostros en una imagen.
def detectFaceOpenCVDnn(net, frame):
    # Crear un blob  de la imagen y aplicar un pre-procesamiento.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Pasar el blob al modelo.
    net.setInput(blob)
    # Obtener la detección.
    detections = net.forward()
    return detections


# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


net = load_model()

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Or use PIL Image (which uses an RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Imagen proporcionada")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("Define el umbral (Threshold)", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Call the face detection model to detect faces in the image.
    detections = detectFaceOpenCVDnn(net, image)

    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold)

    # Display Detected faces.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Imagen resultante")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Descarga la imagen'),
                unsafe_allow_html=True)
