from keras.models import load_model
from scipy.spatial import distance
from tkinter import ttk
from tkinter import *
from PIL import ImageTk
from PIL import Image
import tensorflow as tf
import numpy as np
import threading
import datetime
import argparse
import pickle
import time
import cv2
import os

class FaceRecognition:
    def __init__(self, vid, outputImagePath, outputVideoPath):
        self.vid = vid
        self.outputImagePath = outputImagePath
        self.outputVideoPath = outputVideoPath
        self.frame = None
        self.frame_rect = None
        self.thread = None
        self.stopEvent = None
        self.feature = None
        self.out_video = None
        self.dict_face = {}

        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Load face detection model using OpenCV Haarcascade and Facenet model
        self.faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_alt2.xml")
        self.load_face_db()

        # Create directory to store images and videos
        self.create_dir(outputImagePath)
        self.create_dir(outputVideoPath)

        # Initialize the root window and image panel using Tkinter
        self.root = Tk()
        self.panel = None
        self.content = ttk.Frame(self.root, padding=(3,3,12,12))

        self.lbl_name = ttk.Label(self.content, text="What's your name?")
        self.txt_name = ttk.Entry(self.content)
        self.btn_regist = ttk.Button(self.content, text="Register", command = self.register)
        self.btn_take_pic = ttk.Button(self.content, text="Take Picture", command = self.take_picture)
        self.btn_take_vid = ttk.Button(self.content, text="Start Recording", command = self.record_video)

        self.content.grid(column=0, row=0, sticky=(N, S, E, W))
        self.lbl_name.grid(column=0, row=3, columnspan=2, sticky=(S, W), pady=5, padx=5)
        self.txt_name.grid(column=0, row=4, sticky=(S, E, W), pady=5, padx=5)

        self.btn_regist.grid(column=1, row=4, sticky=(S, E, W), pady=5, padx=5)
        self.btn_take_pic.grid(column=2, row=0, sticky=(N, E, W), pady=5, padx=5)
        self.btn_take_vid.grid(column=2, row=1, sticky=(N, E, W), pady=5, padx=5)

        self.root.columnconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=3)
        self.content.columnconfigure(2, weight=2)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self.content.rowconfigure(2, weight=3)
        self.content.rowconfigure(3, weight=1)
        self.content.rowconfigure(4, weight=1)

        # Start a thread that constantly pools the video sensor for the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.setDaemon(True)
        self.thread.start()

        # Set a callback to handle when the window is closed
        self.root.wm_title("Face Recognition Application")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def video_loop(self):
        # Keep looping over frames until we are instructed to stop
        while not self.stopEvent.is_set():
            self.fps = self.vid.get(cv2.CAP_PROP_FPS)
            # Grab the frame from the video stream, then add haarcascade face detection and 
            # face recognition using Facenet keras models
            _, self.frame = self.vid.read()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = cv2.flip(self.frame, 1)
            self.detect_faces()
            self.frame_rect = self.frame.copy()
            self.image = Image.fromarray(self.frame)
            self.image = ImageTk.PhotoImage(self.image)

            # If the panel is None, we need to initialize it
            if self.panel is None:
                self.panel = ttk.Label(self.content, image=self.image)
                self.panel.image = self.image
                self.panel.grid(column=0, row=0, columnspan=2, rowspan=3, sticky=(N, S, E, W))

            # Otherwise, simply update the panel
            else:
                self.panel.configure(image=self.image)
                self.panel.image = self.image

            if self.out_video is not None:
                self.out_video.write(cv2.cvtColor(self.frame_rect.copy(), cv2.COLOR_RGB2BGR))

    def detect_faces(self):
        rects = self.faceCascade.detectMultiScale(self.frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            self.feature = self.get_embedding(self.frame[y:y+h, x:x+w])
            label = self.most_similar()
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(self.frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), lineType=cv2.LINE_AA)

    def get_embedding(self, crop_face):
        crop_face_rs = cv2.resize(crop_face, (160,160)).reshape(1,160,160,3)
        mean = np.mean(crop_face_rs)
        std = np.std(crop_face_rs)
        std_adj = np.maximum(std, 1.0/np.sqrt(crop_face_rs.size))
        crop_face_rs = np.multiply(np.subtract(crop_face_rs, mean), 1/std_adj)
        with graph.as_default():
            embedding = model.predict(crop_face_rs)
        return embedding
    
    def most_similar(self):
        label = "unknown"
        if self.dict_face:
            dist_list = []
            for key in self.dict_face.keys():
                for arr_face in self.dict_face[key]:
                    dist_list.append([1 - distance.cosine(self.feature, arr_face), key])
            dist_list.sort(key=lambda x: float(x[0]), reverse=True)
            if(dist_list[0][0] > 0.8):
                label = dist_list[0][1] + " " + str(dist_list[0][0]*100)[:2] + "%"
        return label

    def create_dir(self, dir):
        # Create a directory if it's not exists yet
        if not os.path.exists(dir):
            os.mkdir(dir)

    def register(self):
        if self.txt_name.get() in self.dict_face: 
            self.dict_face[self.txt_name.get()].append(self.feature)
        else:
            self.dict_face[self.txt_name.get()] = [self.feature]
        self.txt_name.delete(0, END)

    def load_face_db(self):
        try:
            self.dict_face = pickle.load(open("face_db.pickle", "rb"))
        except:
            print("No such face database")

    def take_picture(self):
        # Grab the current timestamp and use it as the image filename
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        image_path = os.path.sep.join((self.outputImagePath, filename))

        # Save the file
        cv2.imwrite(image_path, cv2.cvtColor(self.frame_rect.copy(), cv2.COLOR_RGB2BGR))
        print("[INFO] Image saved at {}".format(self.outputImagePath))

    def record_video(self):
        if self.btn_take_vid["text"] == "Start Recording":
            self.btn_take_vid["text"] = "Stop Recording"
            ts = datetime.datetime.now()
            filename = "{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            video_path = os.path.sep.join((self.outputVideoPath, filename))
            self.out_video = cv2.VideoWriter(video_path, -1, self.fps, self.size)
        else:
            self.btn_take_vid["text"] = "Start Recording"
            self.out_video.release()
            self.out_video = None
            print("[INFO] Video saved at {}".format(self.outputVideoPath))

    def onClose(self):
        # Save new data to database
        pickle.dump(self.dict_face, open("face_db.pickle", "wb"))

        # set the stop event, cleanup the camera, and allow the rest of the quit process to continue
        print("[INFO] Closing the application...")
        self.stopEvent.set()
        self.vid.release()
        self.root.quit()

if __name__ == "__main__":
    # Initialize a parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-out_i", "--output_image", default="Image",
        help="path to output directory to store images")
    parser.add_argument("-out_v", "--output_video", default="Video",
        help="path to output directory to store videos")
    parser.add_argument("-cam", "--camera_id", type=int, default=0,
        help="Camera ID")
    args = vars(parser.parse_args())

    global graph, model
    graph = tf.get_default_graph()

    model = load_model('model/facenet_keras.h5')
    model.load_weights('model/facenet_keras_weights.h5')

    print("[INFO] Running camera...")
    vid = cv2.VideoCapture(args["camera_id"])

    # Start the app
    faceRecognitionApp = FaceRecognition(vid, args["output_image"], args["output_video"])
    faceRecognitionApp.root.mainloop()