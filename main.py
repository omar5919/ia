import base64
import cv2 as cv
import io
import os
import time
from datetime import datetime

import dlib
import imutils
import numpy as np
import pyshine as ps
from PIL import Image
from bson.json_util import dumps
from engineio.payload import Payload
from flask import Flask, jsonify, flash, redirect
from flask import request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from imutils.video import VideoStream
from pymongo import MongoClient
from werkzeug.utils import secure_filename

connection = "mongodb://tgestiona:Tgestiona2021@143.198.59.116:27017/tgestiona"
rtsp_url = "rtsp://admin:Tgestiona2021@192.168.1.65:554/ch1/main/av_stream"

client = MongoClient(connection)

database = 'tgestiona'
collection = 'deepface'

db = client[database]

dbface = db[collection]
print(db.list_collection_names())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Payload.max_decode_packets = 2048
UPLOAD_FOLDER = 'upload'
ROSTROS_FOLDER = 'rostros'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROSTROS_FOLDER'] = ROSTROS_FOLDER
socketio = SocketIO(app, cors_allowed_origins='*')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_stream = VideoStream(rtsp_url).start()

@app.route("/entrenar", methods=["GET"])
def entrenar():
    labels = []
    facesData = []
    label = 0

    ts = datetime.now()

    # facial_img_paths = []
    # for root, directory, files in os.walk(os.path.dirname(__file__) + "/upload/"):
    #     for file in files:
    #         if '.png' in file:
    #             facial_img_paths.append(root + "/" + file)

    personas = os.listdir(os.path.dirname(__file__) + '/rostros/')
    for p in personas:
        personpath = os.path.dirname(__file__) + '/rostros/' + p
        per = p.split('_')
        labels.append(label)
        facesData.append(cv.imread(personpath, 0))
        label = label + 1
    print('entrenando...')
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write(os.path.dirname(__file__) + '/modelo/modelos.yml')
    return jsonify(ok=400)


@app.route("/obtenercapturas", methods=["GET"])
def obtener_capturas():
    cursor = dbface.find({'tipo': 0})
    list_cur = list(cursor)
    json_data = dumps(list_cur)
    return jsonify(json_data)


@app.route("/obtenerdetecciones", methods=["GET"])
def obtener_detecciones():
    cursor = dbface.find({'tipo': 1})
    list_cur = list(cursor)
    json_data = dumps(list_cur)
    return jsonify(json_data)


@app.route("/subir", methods=["POST"])
def subir():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify(ok=200)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if 'cip' not in request.form:
            return jsonify(ok=300)
        now = datetime.now()
        archivo = '{}_{}_{}.png'.format(request.form.get('usr'), request.form.get('cip'), now.strftime("%d%m%Y%H%M%S"))
        file.filename = '{}_{}_{}.png'.format(request.form.get('usr'), request.form.get('cip'),
                                              now.strftime("%d%m%Y%H%M%S"))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], archivo))

            imread = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], archivo))
            frame = imutils.resize(imread, width=640)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frameaux = frame.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = frameaux[y:y + h, x:x + w]
                rostro = cv.resize(rostro, (150, 150), interpolation=cv.INTER_CUBIC)
                cv.imwrite('{}/'.format(app.config['ROSTROS_FOLDER']) + archivo, rostro)

            f = file.filename.split('_')
            dbface.insert_one({"nombre": f[0], "cip": f[1], "fecha": now, "tipo": 0})
            return jsonify(ok=200)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv.cvtColor(np.array(pimg), cv.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)


global fps, prev_recv_time, cnt, fps_array, fecha_temporal
fps = 30
prev_recv_time = 0
cnt = 0
fps_array = [0]
fecha_temporal = datetime.now()


@socketio.on('image2')
def image2():
    global fps, cnt, prev_recv_time, fps_array, fecha_temporal
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    frame = video_stream.read()
    if frame is None:
        print('No existe frame...')
    else:
        try:
            frame = imutils.resize(frame,width=450)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=10, hspace=10, font_scale=0.5,
                                background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
            imgencode = cv.imencode('.jpeg', frame, [cv.IMWRITE_JPEG_QUALITY, 40])[1]

            # base64 encode
            stringData = base64.b64encode(imgencode).decode('utf-8')
            b64_src = 'data:image/jpeg;base64,'
            stringData = b64_src + stringData
            emit('response_back', stringData)
        except Exception as e:
            cv.putText(frame, 'No existe modelo!', (10, 30), 1, 1.3, (0, 0, 255), 2, cv.LINE_AA)
            imgencode = cv.imencode('.jpeg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])[1]
            stringData = base64.b64encode(imgencode).decode('utf-8')
            b64_src = 'data:image/jpeg;base64,'
            stringData = b64_src + stringData
            emit('response_back', stringData)


@socketio.on('image')
def image(data_image):
    global fps, cnt, prev_recv_time, fps_array, fecha_temporal
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    # frame = (readb64(data_image))
    frame = video_stream.read()
    if frame is None:
        print('No Existe frame...')
    try:
        frame = imutils.resize(frame, width=1280)
        face_recognizer.read(os.path.dirname(__file__) + '/modelo/modelos.yml')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frameaux = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = frameaux[y:y + h, x:x + w]
            rostro = cv.resize(rostro, (150, 150), interpolation=cv.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv.LINE_AA)
            if result[1] < 75:
                cv.putText(frame, 'Registrado', (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                fecha = datetime.now()
                if (fecha - fecha_temporal).seconds > 20:
                    dbface.insert_one({"nombre": "omar loarte", "cip": "1234", "fecha": datetime.now(), "tipo": 1})
                    fecha_temporal = fecha
                    cursor = dbface.find({'tipo': 1})
                    list_cur = list(cursor)
                    json_data = dumps(list_cur)
                    emit('mostrar_detecciones', json_data)

            else:
                cv.putText(frame, 'Desconocido', (x, y - 20), 2,
                           0.8, (0, 0, 255), 1, cv.LINE_AA)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # frame = changeLipstick(frame, [0, 255, 0])
        frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=10, hspace=10, font_scale=0.5,
                            background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
        imgencode = cv.imencode('.jpeg', frame, [cv.IMWRITE_JPEG_QUALITY, 40])[1]

        # base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData

        # emit the frame back
        emit('response_back', stringData)

        fps = 1 / (recv_time - prev_recv_time)
        fps_array.append(fps)
        fps = round(moving_average(np.array(fps_array)), 1)
        prev_recv_time = recv_time
        # print(fps_array)
        cnt += 1
        if cnt == 30:
            fps_array = [fps]
            cnt = 0
    except Exception as e:
        cv.putText(frame, 'No existe modelo!', (10, 30), 1, 1.3, (0, 0, 255), 2, cv.LINE_AA)
        imgencode = cv.imencode('.jpeg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])[1]
        # base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData
        # emit the frame back
        emit('response_back', stringData)
        print("error {}".format(e.__getattribute__("msg")))


def getMaskOfLips(img, points):
    mask = np.zeros_like(img)
    mask = cv.fillPoly(mask, [points], (255, 255, 255))
    return mask


def changeLipstick(img, value):
    img = cv.resize(img, (0, 0), None, 1, 1)
    imgOriginal = img.copy()
    imgColorLips = imgOriginal
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        facial_landmarks = predictor(imgGray, face)
        points = []
        for i in range(68):
            x = facial_landmarks.part(i).x
            y = facial_landmarks.part(i).y
            points.append([x, y])

        points = np.array(points)
        imgLips = getMaskOfLips(img, points[48:61])

        imgColorLips = np.zeros_like(imgLips)

        imgColorLips[:] = value[2], value[1], value[0]
        imgColorLips = cv.bitwise_and(imgLips, imgColorLips)

        value = 1
        value = value // 10
        if value % 2 == 0:
            value += 1
        kernel_size = (6 + value, 6 + value)  # +1 is to avoid 0

        weight = 1
        weight = 0.4 + weight / 400
        imgColorLips = cv.GaussianBlur(imgColorLips, kernel_size, 10)
        imgColorLips = cv.addWeighted(imgOriginal, 1, imgColorLips, weight, 0)

    return imgColorLips


if __name__ == '__main__':
    socketio.run(app, port=9990, debug=False)
