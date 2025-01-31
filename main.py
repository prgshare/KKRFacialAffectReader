import ctypes
import os
import time
import math
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import filedialog
from keras.models import load_model
from keras.layers import Masking
from PIL import Image, ImageTk
from keras.layers import GRU

# A class for OpenFace's FeatureExtraction (FeatureExtraction.dll)
class DLL:
    def __init__(self, dll_path):
        try:
            self.dll = ctypes.cdll.LoadLibrary(dll_path)
            self.begin_feature_extraction = self.dll.begin_feature_extraction
            self.begin_feature_extraction.argtypes = [ctypes.c_wchar_p]
            self.begin_feature_extraction.restype = ctypes.c_int
            self.end_feature_extraction = self.dll.end_feature_extraction
            self.end_feature_extraction.restype = ctypes.c_int
            self.query_au_intensity = self.dll.query_au_intensity
            self.query_au_intensity.argtypes = [ctypes.POINTER(ctypes.c_double)]
            self.query_au_intensity.restype = ctypes.c_bool
            self.set_output_image_size = self.dll.set_output_image_size
            self.set_output_image_size.argtypes = [ctypes.c_int, ctypes.c_int]
            self.get_current_frame = self.dll.get_current_frame
            self.get_current_frame.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            self.get_current_frame.restype = ctypes.c_bool
            self.ready = self.dll.ready
            self.ready.argtypes = []
            self.ready.restype = ctypes.c_bool
            self.estimation_finished = self.dll.estimation_finished
            self.estimation_finished.argtypes = []
            self.estimation_finished.restype = ctypes.c_bool
            self.loaded = True
        except:
            self.loaded = False

# A class for managing the GUI
class Display:
    def __init__(self, dll_file):
        self.au_num = 17 # Number of AUs
        self.font = ('Arial', 16)
        self.running = False
        self.est_num = 0 # Number of estimation

        self.dll = DLL(dll_file)        
        self.model = EstimationModel()

        if not self.dll.loaded:
            self.error = 1
        else:
            self.camera = Camera(self)
            self.graph = Graph(self)
            self.plain = Plain(self)
            self.controller = Controller(self)
            self.root = tk.Tk()
            self.root.title('KKR Facial Affect Reader')
            self.root.geometry('%dx%d+0+0' % (self.graph.width() + self.controller.width(), self.camera.height() + self.graph.height()))
            self.root.protocol('WM_DELETE_WINDOW', self.on_close)
            self.frame_left = tk.Frame(self.root, width=self.graph.width(), height=self.camera.height()+self.graph.height(), bd=0, bg='white')
            self.frame_left.propagate(False)
            self.frame_top = tk.Frame(self.frame_left, width=self.graph.width(), height=self.camera.height(), bd=0, bg='white')
            self.frame_top.propagate(False)
            self.camera.create()
            self.plain.create(width=self.graph.width()-self.camera.width(), height=self.camera.height())
            self.frame_top.pack(anchor=tk.W)
            self.frame_bottom = tk.Frame(self.frame_left, width=self.graph.width(), height=self.graph.height(), bd=0, bg='white')
            self.frame_bottom.propagate(False)
            self.graph.create()
            self.frame_bottom.pack(anchor=tk.W)
            self.frame_left.pack(side=tk.LEFT)
            self.frame_right = tk.Frame(self.root, width=self.controller.width(), height=self.camera.height()+self.graph.height(), bd=0, bg='white')
            self.frame_right.propagate(False)
            self.controller.create()
            self.frame_right.pack(side=tk.RIGHT, anchor=tk.N)
            self.error = 0

    # Execute tkinter's mainloop
    def mainloop(self):
        if self.error == 0:
            self.root.mainloop()
        elif self.error == 1:
            messagebox.showerror('Error', 'Can\'t load library.')

    # This is the function that is called when the "Start" button is pressed.
    def begin(self):
        if self.running:
            return False # Already running
        else:
            if self.model.ready():
                if self.est_num > 0:
                    self.reset()
                self.preprocess()
                return True
            else:
                messagebox.showerror('Error', 'Model file is not loaded.')
                return False

    # Preprocessing
    def preprocess(self):
        if self.controller.is_webcam:
            self.camera.show_message() # If a webcam is used, a message is displayed instructing the user to look into the camera.
        self.root.after(1, self.do_begin)
    
    # This is the function that is called when the estimation starts.
    def do_begin(self):
        # Start the estimation of AUs by calling the DLL function.
        # The argument is either the device id of a webcam or the video file name (corresponding to "-device" and "-f" options of OpenFace's FeatureExtraction)
        if self.dll.begin_feature_extraction(self.controller.get_feature_extraction_arg()) == 0:
            while not self.dll.ready(): # Check if the DLL is ready for detection of AUs.
                time.sleep(0.1) # Wait until the DLL is ready for detection.

            self.running = True
            self.t = 0
            self.aus = np.zeros(self.au_num, dtype=np.float64)
            self.au_seq = np.array([self.aus])
            self.valence = []
            self.arousal = []
            #print('Collecting %d frames ...' % self.window_size)
            self.root.after(100, self.update)
        else:
            messagebox.showerror('Error', 'Initialization failed.')
        
    # Update current status
    def update(self):
        if self.running:
            if self.dll.query_au_intensity(self.aus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))): # Fetch the intensity values of AUs.
                if len(self.au_seq) < self.window_size:
                    # Estimation of valence and arousal begins after collecting designated number of frames (i.e., self.window_size).
                    self.au_seq = np.append(self.au_seq, [self.aus], axis=0)
                    if not self.controller.is_webcam:
                        self.camera.update() # Display current frame obtained from a webcam.
                else:
                        if self.t == 0:
                            e = 0
                            self.t = time.perf_counter()
                            self.camera.hide_message()
                        else:
                            e = math.floor((time.perf_counter() - self.t) * 1000)
                        self.au_seq = np.append(np.delete(self.au_seq, 0, 0), [self.aus], axis=0)
                        # Estimate the instensity of valence and arousal.
                        predicted = self.model.predict(np.asarray(self.au_seq, dtype=np.float64).reshape(1, self.window_size, self.au_num))
                        #if len(self.valence) == 0:
                        #    print('Time(ms),Valence,Arousal')
                        #print('%08d,%.5f,%.5f' % (e, predicted[0][0], predicted[0][1]))
                        self.valence.append(predicted[0][0])
                        self.arousal.append(predicted[0][1])
                        self.camera.update() # Display current frame obtained from a webcam.
                        self.plain.update(predicted[0][0], predicted[0][1]) # Display current V/A plane.
                        self.graph.update(predicted) # Update the graph.
            if (not self.controller.is_webcam) and self.dll.estimation_finished():
                self.controller.video_start_stop() # In case of using a video file, the estimation ends when all the frames are processed.
            else:
                self.root.after(1, self.update) # In case of using a webcam, the estimation continues until the "Stop" button is pressed.

    # Finalization
    def end(self):
        if self.running:
            self.running = False
            #print('Average estimation speed: %.2fFPS' % (1 / ((time.perf_counter() - self.t) / len(self.valence))))
            self.dll.end_feature_extraction()
            self.est_num = self.est_num + 1
            return True
        else:
            return False
    
    # Reset
    def reset(self):
        self.camera.reset()
        self.graph.reset()
        self.plain.reset()

    # This function is used for drawing the graph.
    def get_va(self):
        return self.valence, self.arousal
    
    # Load an estimation model file
    def load_model(self, model_file_name):
        if self.model.update_model(model_file_name):
            self.window_size = self.model.get_window_size()

    # Confirmation for closing the window
    def on_close(self):
        if messagebox.askokcancel('Confirmation', 'Do you want to close the window?'):
            self.root.destroy()

# A class to handle an estimation model
class EstimationModel:
    def __init__(self):
        self.model = None

    # Obtain the estimated instensity values of valence and arousal.
    def predict(self, au_seq):
        predicted = self.model.predict(au_seq, batch_size=1, verbose=0)
        # The instenstiy value of valence and arousal ranges from 1 to 5 in this system while the predicted values are not limited to this range.
        for i in range(2):
            if predicted[0][i] < 1:
                predicted[0][i] = 1
            elif predicted[0][i] > 5:
                predicted[0][i] = 5
        return predicted

    # Load an estimation model file.
    def update_model(self, model_file):
        try:
            self.model = load_model(model_file)
            #print('Model file %s loaded.' % model_file)
        except:
            messagebox.showerror('Error', 'Can\'t load the model file.')
            self.model = None
            return False
        return True
    
    # Detect the window_size from the loaded model file.
    def get_window_size(self):
        return self.model.input_shape[1]
    
    # Check if ready for estimation
    def ready(self):
        return self.model != None
        
# A class to operate a webcam
class Camera:
    def __init__(self, owner):
        self.owner = owner
        self.image_width = 640
        self.image_height = 480
        owner.dll.set_output_image_size(self.image_width, self.image_height)

    # Create a region to display an image captured by the webcam.
    def create(self):
        if hasattr(self, 'image'):
            return True
        if not hasattr(self.owner, 'frame_top'):
            return False
        self.image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8) # This is used to store the image data.
        self.canvas_camera = tk.Canvas(self.owner.frame_top, width=self.image_width, height=self.image_height, bd=0, highlightthickness=0)
        self.canvas_camera.pack(side=tk.LEFT)
        self.camera_update_num = 0
        return True

    # Displays a message indicating that preparations are in progress.
    def show_message(self):
        self.canvas_camera.create_text(self.image_width // 2, self.image_height // 2, text='The system is in preparation...', font=self.owner.font, tag='prepare')

    # Hide the message.
    def hide_message(self):
        self.canvas_camera.delete('prepare')

    # Update the image from the webcam.
    def update(self):
        # Retrieve the image data of the current frame from the webcam.
        if self.owner.dll.get_current_frame(self.image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), self.image.strides[0], self.image.strides[1], True):
            # If the image is retrieved, display it.
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            if self.camera_update_num > 0:
                self.canvas_camera.delete('camera')
            self.camera_update_num = self.camera_update_num + 1
            self.canvas_camera.create_image(0, 0, image=self.tk_image, anchor=tk.NW, tag='camera')

    # Reset
    def reset(self):
        self.canvas_camera.delete('camera')
        self.camera_update_num = 0

    # Returns the width of an image
    def width(self):
        return self.image_width
    
    # Returns the height of an image
    def height(self):
        return self.image_height

# A class to display a valence/arousal (V/A) plane
class Plain:
    def __init__(self, owner):
        self.owner = owner
        self.prev_focus = -1

    # Create the plane.
    def create(self, width=0, height=0):
        if not hasattr(self.owner, 'frame_top'):
            return False
        if hasattr(self, 'canvas_plain'):
            return True
        if width > 0:
            self._width = width
        else:
            self._width = 400
        if height > 0:
            self._height = height
        else:
            self._height = 400
        self.canvas_plain = tk.Canvas(self.owner.frame_top, width=self.width(), height=self.height(), bg='white', bd=0, highlightthickness=0)
        self.canvas_plain.pack(side=tk.RIGHT)
        block_height = self.height() // 7
        #block_width = self.width() // 7
        block_width = block_height
        origin_x = block_width
        origin_y = block_height * 6

        self.panels = []
        for i in range(5):
            y = origin_y - block_height * (5 - i)
            for j in range(5):
                x = origin_x + block_width * j
                tag = 'p%d' % (i * 5 + j)
                self.panels.append(self.canvas_plain.create_rectangle(x, y, x + block_width, y + block_height, fill='white', width=0, tag=tag))
        
        self.line_x_axes = []
        self.line_y_axes = []
        self.x_labels = []
        self.y_labels = []
        for i in range(5):
            y = origin_y - block_height * (i + 1)
            self.line_x_axes.append(self.canvas_plain.create_line(origin_x, y, origin_x + block_width * 5, y, fill='gray85', width=1))
            x = origin_x + block_width * (i + 1)
            self.line_y_axes.append(self.canvas_plain.create_line(x, origin_y, x, origin_y - block_height * 5, fill='gray85', width=1))
            text = '%d' % (i + 1)
            self.x_labels.append(self.canvas_plain.create_text(origin_x + block_width * i + block_width // 2, origin_y + 16, text=text, font=self.owner.font))
            self.y_labels.append(self.canvas_plain.create_text(origin_x - 16, origin_y - block_height * i - block_height // 2, text=text, font=self.owner.font))
        self.line_x_axis = self.canvas_plain.create_line(origin_x, origin_y, origin_x + block_width * 5 + 20, origin_y, arrow=tk.LAST, fill='black', width=2)
        self.line_y_axis = self.canvas_plain.create_line(origin_x, origin_y, origin_x, origin_y - block_height * 5 - 20, arrow=tk.LAST, fill='black', width=2)
        self.label_valence = self.canvas_plain.create_text(origin_x + block_width * 5 + 60, origin_y, text='Valence', font=self.owner.font)
        self.label_arousal = self.canvas_plain.create_text(origin_x, origin_y - block_height * 5 - 30, text='Arousal', font=self.owner.font)

        self.point = self.canvas_plain.create_oval(origin_x - 3, origin_y - 3, origin_x + 3, origin_y + 3, fill='red', width=0, state=tk.HIDDEN, tag='point')

        self.block_width = block_width
        self.block_height = block_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        return True

    # Update the plane.
    def update(self, valence, arousal):
        # Update the point.
        point_x = self.origin_x + math.floor(self.block_width * (valence - 0.5))
        point_y = self.origin_y - math.floor(self.block_height * (arousal - 0.5))
        self.canvas_plain.coords('point', point_x - 3, point_y - 3, point_x + 3, point_y + 3)

        # Update the background color.
        if self.prev_focus < 0:
            self.canvas_plain.itemconfigure('point', state=tk.NORMAL)
        if valence < 4.5:
            focus_x = math.floor(valence - 0.5)
        else:
            focus_x = 4
        if arousal < 4.5:
            focus_y = 4 - math.floor(arousal - 0.5)
        else:
            focus_y = 0
        focus = focus_x + focus_y * 5
        if focus != self.prev_focus:
            if self.prev_focus >= 0:
                tag = 'p%d' % self.prev_focus
                self.canvas_plain.itemconfigure(tag, fill='white')
            tag = 'p%d' % focus
            self.canvas_plain.itemconfigure(tag, fill='misty rose')
            self.prev_focus = focus

    # Reset
    def reset(self):
        self.canvas_plain.itemconfigure('point', state=tk.HIDDEN)
        for i in range(25):
            tag = 'p%d' % i
            self.canvas_plain.itemconfigure(tag, fill='white')
        self.prev_focus = -1

    # Returns the width of the plane.
    def width(self):
        return self._width
    
    # Returns the height of the plane.
    def height(self):
        return self._height

# A class to display a graph
class Graph:
    def __init__(self, owner):
        self.owner = owner
        self.x_axis_width = 50
        self.graph_width = 1000
        self.legend_width = 125
        self.margin_t = 30
        self.y_axis_height = 50
        self.graph_height = 200
        self.frames = 100

    # Create the graph
    def create(self):
        if hasattr(self, 'canvas_graph'):
            return True
        if not hasattr(self.owner, 'frame_bottom'):
            return False
        self.canvas_graph = tk.Canvas(self.owner.frame_bottom, width=self.width(), height=self.height(), bd=0, highlightthickness=0, bg='white')
        self.canvas_graph.pack(side=tk.BOTTOM)

        origin_x = self.x_axis_width
        origin_y = self.margin_t + self.graph_height
        self.x_labels = []
        for i in range(10):
            text = '%d' % ((i + 1) * 10)
            tag = 'label_x%d' % i
            self.x_labels.append(self.canvas_graph.create_text(origin_x + self.graph_width * (i + 1) // 10, self.margin_t + self.graph_height + 16, text=text, font=self.owner.font, tag=tag))
        self.y_labels = []
        for i in range(5):
            text = '%d' % (i + 1)
            self.y_labels.append(self.canvas_graph.create_text(origin_x - 16, origin_y - self.graph_height * i // 4, text=text, font=self.owner.font))
        self.line_x_axes = []
        for i in range(10):
            tag = 'axis_x%d' % i
            x = origin_x + self.graph_width * (i + 1) // 10
            self.line_x_axes.append(self.canvas_graph.create_line(x, origin_y, x, self.margin_t, fill='gray85', width=1, tag=tag))
        self.line_y_axes = []
        for i in range(4):
            y = origin_y - self.graph_height * (i + 1) // 4
            self.line_y_axes.append(self.canvas_graph.create_line(origin_x, y, origin_x + self.graph_width, y, fill='gray85', width=1))
        self.line_x_axis = self.canvas_graph.create_line(origin_x, origin_y, origin_x, self.margin_t - 20, arrow=tk.LAST, fill='black', width=2)
        self.line_y_axis = self.canvas_graph.create_line(origin_x, origin_y, origin_x + self.graph_width + 20, origin_y, arrow=tk.LAST, fill='black', width=2)
        self.title_x = self.canvas_graph.create_text(origin_x + self.graph_width // 2, origin_y + 34, text='Frame', font=self.owner.font)
        
        x = self.x_axis_width + self.graph_width + 10
        self.legend_line_valence = self.canvas_graph.create_line(x, self.margin_t, x + 24, self.margin_t, fill='blue', width=1)
        self.legend_line_arousal = self.canvas_graph.create_line(x, self.margin_t + 20, x + 24, self.margin_t + 20, fill='red', width=1)
        self.legend_label_valence = self.canvas_graph.create_text(x + 28, self.margin_t, text='Valence', anchor='w', font=self.owner.font)
        self.legend_label_arousal = self.canvas_graph.create_text(x + 28, self.margin_t + 20, text='Arousal', anchor='w', font=self.owner.font)
        
        self.va = None
        self.current_frame = 0
        return True
    
    # Update the graph.
    def update(self, predicted):
        if self.va is None:
            self.va = np.array(predicted)
        elif self.va.shape[0] < self.frames:
            self.va = np.append(self.va, predicted, axis=0)
        else:
            self.va = np.append(np.delete(self.va, 0, 0), predicted, axis=0)
        self.current_frame = self.current_frame + 1
        if self.va.shape[0] >= 2:
            self.draw_graph()

    # Draw the graph.
    def draw_graph(self):
        valence = []
        arousal = []
        for i in range(self.va.shape[0]):
            x = self.x_axis_width + i * (self.graph_width - 1) // (self.frames - 1)
            valence.append(x)
            valence.append(math.floor((5.0 - self.va[i][0]) * (self.graph_height - 1) / 4.0) + self.margin_t)
            arousal.append(x)
            arousal.append(math.floor((5.0 - self.va[i][1]) * (self.graph_height - 1) / 4.0) + self.margin_t)
        if len(valence) == 4:
            self.line_valence = self.canvas_graph.create_line(valence[0], valence[1], valence[2], valence[3], fill='blue', tag='valence')
            self.line_arousal = self.canvas_graph.create_line(arousal[0], arousal[1], arousal[2], arousal[3], fill='red', tag='arousal')
        else:
            self.canvas_graph.coords('valence', *valence)
            self.canvas_graph.coords('arousal', *arousal)

        # The following code is used for scrolling.
        if self.current_frame > self.frames:
            width_unit = self.graph_width / 10
            frames_per_unit = self.frames // 10
            r = self.current_frame % frames_per_unit
            shift = width_unit * r / (frames_per_unit - 1)
            origin_y = self.margin_t + self.graph_height
            xlabel_val = (self.current_frame + frames_per_unit - self.frames) // frames_per_unit * frames_per_unit
            for i in range(10):
                x = self.x_axis_width + self.graph_width * (i + 1) // 10 - shift
                self.canvas_graph.coords('axis_x%d' % i, x, origin_y, x, self.margin_t)
                text = '%d' % xlabel_val
                self.canvas_graph.itemconfigure('label_x%d' % i, text=text)
                xlabel_val = xlabel_val + frames_per_unit
                self.canvas_graph.coords('label_x%d' % i, x, origin_y + 16)

    # Reset
    def reset(self):
        self.canvas_graph.delete('valence')
        self.canvas_graph.delete('arousal')
        origin_x = self.x_axis_width
        origin_y = self.margin_t + self.graph_height
        for i in range(10):
            x = origin_x + self.graph_width * (i + 1) // 10
            self.canvas_graph.coords('axis_x%d' % i, x, origin_y, x, self.margin_t)
            text = '%d' % ((i + 1) * 10)
            self.canvas_graph.itemconfigure('label_x%d' % i, text=text)
            self.canvas_graph.coords('label_x%d' % i, x, origin_y + 16)
        self.va = None
        self.current_frame = 0

    # Returns the width of the graph.
    def width(self):
        return self.x_axis_width + self.graph_width + self.legend_width
    
    # Returns the height of the graph.
    def height(self):
        return self.y_axis_height + self.graph_height + self.margin_t

# A class to handle buttons
class Controller:
    def __init__(self, owner):
        self.owner = owner
        self._width = 100
        self.font = ('Arial', 10)

    # Create the buttons.
    def create(self):
        if not hasattr(self.owner, 'frame_right'):
            return False
        if hasattr(self, 'webcam_button'):
            return True
        self.model_button = tk.Button(self.owner.frame_right, text='Load Model', command=self.load_model, font=self.font, width=8, height=2)
        self.model_button.pack()
        self.webcam_button_text = tk.StringVar()
        self.webcam_button_text.set('Start')
        self.webcam_button = tk.Button(self.owner.frame_right, textvariable=self.webcam_button_text, command=self.webcam_start_stop, font=self.font, width=8, height=2)
        self.webcam_button.pack()
        self.video_button_text = tk.StringVar()
        self.video_button_text.set('Load Video')
        self.video_button = tk.Button(self.owner.frame_right, textvariable=self.video_button_text, command=self.video_start_stop, font=self.font, width=8, height=2)
        self.video_button.pack()
        self.output_button = tk.Button(self.owner.frame_right, text='Show Graph', command=self.show_graph, font=self.font, width=8, height=2)
        self.output_button.pack()
        self.output_button['state'] = 'disabled'
        return True
    
    # Load a model file.
    def load_model(self):
        if not self.owner.running:
            modelfile = filedialog.askopenfilename(title='Open a model file', filetypes=[('Model file', '*.*')])
            if modelfile:
                self.owner.load_model(modelfile)

    # Switch "Start" and "Stop" for a webcam.
    def webcam_start_stop(self):
        self.is_webcam = True
        self.start_stop()

    # Switch "Start" and "Stop" for a video file.
    def video_start_stop(self):
        if self.owner.running:
            self.start_stop()
        else:
            self.videofile = filedialog.askopenfilename(title='Open a video file', filetypes=[('MP4 file(*.mp4)', '.mp4')])
            if self.videofile:
                self.is_webcam = False
                self.start_stop()

    # Switch "Start" and "Stop"
    def start_stop(self):
        if self.owner.running:
            # Stop
            if self.owner.end():
                if self.is_webcam:
                    self.webcam_button_text.set('Start')
                    self.model_button['state'] = 'normal'
                    self.video_button['state'] = 'normal'
                else:
                    self.video_button_text.set('Load Video')
                    self.model_button['state'] = 'normal'
                    self.webcam_button['state'] = 'normal'
                self.output_button['state'] = 'normal'
        else:
            # Start
            if self.owner.begin():
                if self.is_webcam:
                    self.webcam_button_text.set('Stop')
                    self.model_button['state'] = 'disabled'
                    self.video_button['state'] = 'disabled'
                else:
                    self.video_button_text.set('Stop')
                    self.model_button['state'] = 'disabled'
                    self.webcam_button['state'] = 'disabled'
                self.output_button['state'] = 'disabled'

    # Show a graph of the estimation result.
    def show_graph(self):
        if not self.owner.running:
            valence, arousal = self.owner.get_va()
            x = np.arange(1, len(valence) + 1)
            valence = np.array(valence)
            arousal = np.array(arousal)

            fig = plt.figure('Estimation result')

            # Create a line graph.
            sp1 = fig.add_subplot(2, 1, 1)
            sp1.plot(x, valence, color='blue', label='Valence')
            sp1.plot(x, arousal, color='red', label='Arousal')
            sp1.set_xlabel('Frame')
            sp1.set_ylabel('Valence/Arousal')
            sp1.legend()

            # Create a graph showing the distribution of the frequencey of V/A intensity.
            sp2 = fig.add_subplot(2, 1, 2)
            colors = self.get_colors(valence, arousal)
            sp2.scatter(np.array([1]), np.array([1]))
            sp2.set_xlabel('Valence')
            sp2.set_ylabel('Arousal')
            sp2.set_aspect('equal')
            sp2.set_xlim(0.5, 5.5)
            sp2.set_ylim(0.5, 5.5)
            sp2.set_xticks(np.arange(1, 6))
            sp2.set_yticks(np.arange(1, 6))
            for i in range(5):
                for j in range(5):
                    sp2.axhspan(i + 0.5, i + 1.5, j / 5, (j + 1) / 5, color=colors[i][j], alpha=1)
            
            fig.tight_layout()
            fig.show()

    # Returns the background color for the V/A graph.
    def get_colors(self, valence, arousal):
        freq = np.zeros((5, 5))
        for i in range(valence.shape[0]):
            v = math.floor(valence[i] - 0.5)
            a = math.floor(arousal[i] - 0.5)
            freq[a][v] = freq[a][v] + 1
        colors = []
        for i in range(5):
            colors.append([])
            for j in range(5):
                intensity = freq[i][j] / valence.shape[0]
                if intensity > 0.5:
                    intensity = 0.5 # Red is used for the frequency more than 50%.
                intensity = 255 - math.floor(intensity * 510)
                colors[i].append('#ff%02x%02x' % (intensity, intensity))
        return colors
    
    # Returns the width.
    def width(self):
        return self._width

    # Returns the argument for 'begin_feature_extraction'
    def get_feature_extraction_arg(self):
        if self.is_webcam:
            return '0'
        return self.videofile

if __name__ == '__main__':
    display = Display(os.getcwd() + '/FeatureExtraction.dll')
    display.mainloop()
