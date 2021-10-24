from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os


class Application:
    def __init__(self, output_path="./"):
        self.vs = cv2.VideoCapture(0)  # levanta la camara
        self.output_path = output_path  # ruta donde guardar las imagenes
        self.current_image = None  # imagen de la camara

        self.root = tk.Tk()  # tkinter
        self.root.title("VMS")
        self.root.iconphoto(False, tk.PhotoImage(file='./ico.png'))
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

        self.root_frame = None

        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.ubicacion_faces = os.path.dirname(__file__) + '/faces/'

        # BOTON DE CAPTURAR
        btn = tk.Button(self.root, text="Capturar!", command=self.capturar)
        btn.pack(fill="both", expand=True, padx=10, pady=10)

        # BOTON DE ENTRENAR
        btn_entrenar = tk.Button(self.root, text="Entrenar!", command=self.entrenar)
        btn_entrenar.pack(fill="both", expand=True, padx=10, pady=10)

        # BOTON DE RECONOCER
        btn_reconocer = tk.Button(self.root, text="Reconocer!", command=self.reconocer)
        btn_reconocer.pack(fill="both", expand=True, padx=10, pady=10)

        # inicia la lectura de la camara
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()  # leer cada frame
        if ok:  # loop
            self.root_frame = frame.copy()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
        self.root.after(30, self.video_loop)

    def capturar(self):
        gray = cv2.cvtColor(self.root_frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceClassif.detectMultiScale(gray, 1.3, 5)
        count = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(self.root_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = self.root_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(self.ubicacion_faces + '/rotro_{}.jpg'.format(count), rostro)
            count = count + 1
        # ts = datetime.datetime.now()  # obtiene la fechahora actual
        # filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # el archivo
        # p = os.path.join(self.output_path, filename)  # ruta
        # self.current_image.save(p, "JPEG")  # guarda la imagen
        # print("[INFO] saved {}".format(filename))

    def entrenar(self):
        ts = datetime.datetime.now()  # obtiene la fechahora actual
        print('entrenando...')

    def reconocer(self):
        ts = datetime.datetime.now()  # obtiene la fechahora actual
        print('reconociendo...')

    def destructor(self):
        print("[INFO] cerrando...")
        self.root.destroy()
        self.vs.release()  # liberar la camara
        cv2.destroyAllWindows()


# argumentos al frm
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./", help="ruta donde se almacena las imagenes")
args = vars(ap.parse_args())

# inicia vms
print("[INFO] iniciando...")
pba = Application(args["output"])
pba.root.mainloop()
