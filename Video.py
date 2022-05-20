# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

import threading

def canny1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # su dung bo loc canh canny voi nguong tu 50 den 150
    canny = cv2.Canny(blur, 50, 150)
    return canny


def segment1(frame):
    #height = frame.shape[0]
    # tao hinh tam giac (goc trai, goc phai, goc tren cung)
    polygons = np.array([
        [(0, 110), (256, 110), (125, 40)] #6.mp4
        # [(0, height), (800, height), (380, 290)]
    ])
    # tao anh co kich thuong giong frame va co cac pixel = 0
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    segment = cv2.bitwise_and(frame, mask)
    return segment


def find_lines(frame, lines):
    left = []
    right = []

    for line in lines:
        # x1, y1, x2, y2 la toa do cua 2 line duong
        x1, y1, x2, y2 = line.reshape(4)
        # Tim slope and y-intercept cua 2 vach ke duong
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # neu slope < 0, vach ke duong nam ben trai, va nguoc lai
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # lay gia tri trung binh cua slope va intercept
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    # Tim toa do x1, y1, x2, y2 cua vach trai va vach phai
    left_line = lines_coordinates(frame, left_avg)
    right_line = lines_coordinates(frame, right_avg)
    return np.array([left_line, right_line])


def lines_coordinates(frame, parameters):
    slope, intercept = parameters

    y1 = frame.shape[0]
    # sets y2 co toa do cach y1 80 pixel
    y2 = int(y1 - 80)
    # Tim toa do x1 voi phuong trinh (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Tim toa do x2 voi phuong trinh (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    # print(x1, y1)
    # print(x2, y2)
    return np.array([x1, y1, x2, y2])


def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # ve line giua 2 toa do, voi mau xanh va do lon = 2 pixel
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return lines_visualize


def lineFromPoints(P, Q, Y):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])  # c = ax + by

    # tu c = ax + by
    x = (c - b * Y) / a  # Y la chieu cao cua 2 diem xet turn left, right
    return x


def turn_predict(image_center, right_lane_pos, left_lane_pos):
    # lane_center = left_lane_pos + (right_lane_pos - left_lane_pos) / 2
    lane_center = (left_lane_pos + right_lane_pos) / 2

    if (lane_center - image_center < -10):
        return ("Turning left")
    elif (lane_center - image_center < 10):
        return ("straight")
    else:
        return ("Turning right")


def tonghop(image):
    frame = image
    # result = cv2.imread(path)
    # Y = 30  #1.mp4
    Y = 75  # 6.mp4,
    # 2.mp4
    image_center = 128

    canny = canny1(frame)
    # cv2.imshow('ad',canny)
    # plt.imshow(canny)
    # plt.show()
    segment = segment1(canny)
    # cv2.imshow('ad',segment)
    # plt.imshow(segment)
    #plt.show()

    #hough = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    hough = cv2.HoughLinesP(segment, 3, np.pi / 180, 50, 50, minLineLength=50, maxLineGap=50)
    # Tim toa do line trai va phai bang cach tinh trung binh slope va intercept cua cac line trai hoac phai
    # print('hou', hough)
    # print("type hough", type(hough))
    lines = find_lines(frame, hough)

    P = lines[0][0:2]
    Q = lines[0][2:4]
    P1 = lines[1][0:2]
    Q1 = lines[1][2:4]
    #print(P, Q, P1, Q1)
    left_lane_pos = lineFromPoints(P, Q, Y)
    # print("left_pos", left_lane_pos)
    right_lane_pos = lineFromPoints(P1, Q1, Y)
    # print("right_pos", right_lane_pos)
    turnpredict = turn_predict(image_center, right_lane_pos, left_lane_pos)
    # print("turnpredict", turnpredict)

    # Visualizes the lines
    lines_visualize = visualize_lines(frame, lines)

    lines_visualize = cv2.resize(lines_visualize, (512, 256))
    frame = cv2.resize(frame, (512, 256))

    lines_visualize = cv2.circle(lines_visualize, (int(left_lane_pos*2), Y*2), radius=0, color=(0, 0, 255), thickness=8*2)
    lines_visualize = cv2.circle(lines_visualize, (int(right_lane_pos*2), Y*2), radius=0, color=(0, 0, 255), thickness=8*2)
    # cv2_imshow(lines_visualize)
    # cv2.imwrite("./lines_visualize.jpg", lines_visualize);

    # Tron frame va lines_visualize lai voi nhau
    result = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    # result = cv2.resize(result, (512, 256))
    cv2.putText(result, turnpredict, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos) / 2
    # print(lane_center)
    # print(int(lane_center))

    result = cv2.circle(result, (int(lane_center*2), Y*2), radius=0, color=(0, 255, 0),
                        thickness=8*2)
    result = cv2.circle(result, (int(image_center*2), Y*2), radius=0, color=(255, 0, 0),
                        thickness=8*2)


    # cv2.imwrite("./output.jpg", output);
    drive_angle = math.degrees(math.atan(lane_center - image_center) / (128 - Y))
    # drive_angle = round(drive_angle, 6)
    # print("drive_angle", drive_angle)
    cv2.putText(result, str(drive_angle), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Opens a new window and displays the output frame
    # cv2.imshow(output)
    # cv2_imshow(result)
    # cv2.imwrite("./result.jpg", result);

    # right_lane_pos, left_lane_pos vị trí x1, x2  tìm được ở trước đó

    return result

def select_video():
    global path
    path = filedialog.askopenfilename()
    Label(text='Opened file: ', ).place(relx=0.21, rely=0.75, anchor='e')
    Label(text=path).place(relx=0.22, rely=0.75, anchor='w')
    return path

def view_frame():
    # grab a reference to the image panels
    global panelA, panelB
    global path
    # open a file chooser dialog and allow the user to select an input
    # image
    #path = filedialog.askopenfilename()
    #path = select_video()
    #path = '6.mp4'
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        i = 0
        cap = cv2.VideoCapture(path)

        if cap.isOpened():
            ret, frame = cap.read()
        else:
            ret = False
        g =0
        while ret:
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video

            ret, frame = cap.read()
            if i == 0:
                start = time.time()
                frame = cv2.resize(frame, (256, 128), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                image = frame
                # cv2.imshow('af', frame)
                # image = cv2.imread(frame)
                result = tonghop(frame)

                image = cv2.resize(image, (512, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                # result = cv2.resize(result, (512, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                # cv2.imshow('a2', result)
                # OpenCV represents images in BGR order; however PIL represents
                # images in RGB order, so we need to swap the channels
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                # convert the images to PIL format...
                image = Image.fromarray(image)
                result = Image.fromarray(result)
                # ...and then to ImageTk format
                image = ImageTk.PhotoImage(image)
                result = ImageTk.PhotoImage(result)
                # if the panels are None, initialize them
                panelA.config(image=image)
                panelA.img = image
                # while the second panel will store the edge map
                panelB.config(image=result)
                panelB.img = result
                g=g+1
                print(g)
                end = time.time()
                print('time', start - end)
                if stop == True:
                    cap.release()
                    break
            i = i + 1
            if i == 10:
                i = 0
        #         time.sleep(3)
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        #     i = i +1
    # The following frees up resources and closes all windows
            cv2.waitKey(5)
        cap.release()
        # cv2.destroyAllWindows()

def play():
    global stop
    stop = False
    t = threading.Thread(target=view_frame)
    t.start()

def stop1():
    global stop
    stop = True

# initialize the window toolkit along with the two image panels
root = Tk()
root.title("Ung Dung Ho Tro Dieu Huong Xe")
root.geometry("1100x600")
panelA = None
panelB = None
stop = None

Label(text='Ung Dung Ho Tro Dieu Huong Xe', font=('Aerial 15 bold')).place(relx=0.5, rely=0.0, anchor='n')

# root.grid_columnconfigure(1, weight=1)

relaframe = Frame(bg='black')
relaframe.place(relx=0.5, rely=0.2, height=256, width=1024, anchor='n')
panelA = Label(relaframe, bg='black')
panelA.place(relx=0.5, rely=0.5, height=256, width=512, anchor='e')

panelB = Label(relaframe, bg='black')
panelB.place(relx=0.5, rely=0.5, height=256, width=512, anchor='w')

Button(text='Open', command=select_video).place(relx=0.12, rely=0.75, height=50, width=100, anchor='e')
Button(text='Play', command=play).place(relx=0.12, rely=0.85, height=50, width=100, anchor='e')
Button(text='Stop', command=stop1).place(relx=0.15, rely=0.85, height=50, width=100, anchor='w')


# Label(text='Ung Dung Ho Tro Dieu Huong Xe', font=('Aerial 15 bold'), justify='center').pack()
#
# relaframe = Frame(bg='black')
# relaframe.pack()
# panelA = Label(relaframe, bg='black')
# panelA.pack(side="left", padx=10, pady=10)
#
# panelB = Label(relaframe, bg='black')
# panelB.pack(side="right", padx=10, pady=10)
#
# # create a button, then when pressed, will trigger a file chooser
# # dialog and allow the user to select an input image; then add the
# # button the GUI
#
# # btn = Button(root, text="Select an image", command=select_image)
# # btn.pack(side="bottom", fill="both", padx="10", pady="10")
#
# Button(text='Open', command=play).pack(side="left", fill="x", padx="10", pady="10")
# Button(text='Play', command=play).pack(side="left", fill="x", padx="10", pady="10")
# Button(text='Stop', command=stop1).pack(side="left", fill="x", padx="10", pady="10")
# Button(root, text='Play', command=play).grid(row=5, column=5, sticky=E)

# kick off the GUI
root.mainloop()
