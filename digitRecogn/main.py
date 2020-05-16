import scipy.io
import matplotlib.pyplot as pt
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
from numpy import array, exp
import pprint


pp = pprint.PrettyPrinter(indent=20)
np.set_printoptions(linewidth=np.inf)


# Iniatialization
digits = scipy.io.loadmat('ex3data1.mat')
weights = scipy.io.loadmat('ex3weights.mat')
digitsMatrix = digits['X']
digitsIndex = digits['y']
theta1 = weights['Theta1']
theta2 = weights['Theta2']
myDigit = []


def predict(data, theta1, theta2):

    m = len(data)
    num_labels = len(theta2)

    addOn = np.ones(m)
    a1 = np.column_stack((addOn, data))

    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)


    addOn2 = np.ones(len(a2))
    a2 = np.column_stack((addOn2, a2))

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    maximun = max(a3[0])
    index = np.where(a3[0] == maximun)
    realIndex = index[0][0] + 1
    if (realIndex == 10):
        realIndex = 0
    return (realIndex, maximun)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class App(tk.Tk):
    def  __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.my_canvas = tk.Canvas(self, width=400, height=400, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_regonize = tk.Button(self, text = "Recognize", command = self.recognize)
        self.my_canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_regonize.grid(row=1, column=1, pady=2, padx=2)
        self.my_canvas.bind('<B1-Motion>', self.paint)

    def clear_all(self):
        self.my_canvas.delete("all")
        self.label.configure(text= 'draw')

    def recognize(self):
        my_canvas_id = self.my_canvas.winfo_id()
        rect = win32gui.GetWindowRect(my_canvas_id)
        newRect = ()
        newRect = (rect[0] + 2, rect[1] + 2, rect[2] - 2, rect[3] - 2)
        im = ImageGrab.grab(newRect)
        img = im.convert('L')
        img = img.resize((20, 20))
        img = np.array(img)
        img = 1 - (img / 255)
        myDigit = img.flatten()
        myDigit = myDigit.reshape(1, 400)
        index, max = predict(myDigit, theta1, theta2)
        self.label.configure(text= str(index)+', '+ str(int(max*100))+'%')

    def paint(self, event):
        color='black'
        x1, y1 = (event.x-1), (event.y-1)
        x2, y2 = (event.x+1), (event.y+1)
        r=8
        self.my_canvas.create_oval(x1 - r, y1 - r, x2 + r, y2 + r, fill=color, outline=color)

# show one of the pic  from the data set for comparaison
# lol = digitsMatrix[0].reshape(20, 20)
# pt.imshow(lol, cmap='gray')
# pt.show()

#print(digitsMatrix[2500])
# print(digitsIndex[2500])

app = App()
mainloop()
