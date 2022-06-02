import time
#import cv2
import numpy as np
import jetson.inference
import jetson.utils
import cv2
#import threading 
################################----моторы
import board
import digitalio

redcubled = digitalio.DigitalInOut(board.D26) #red
greencubled = digitalio.DigitalInOut(board.D20) #green


redcubled.direction = digitalio.Direction.OUTPUT
greencubled.direction = digitalio.Direction.OUTPUT

redled = digitalio.DigitalInOut(board.D13)
redled.direction = digitalio.Direction.OUTPUT

greenled = digitalio.DigitalInOut(board.D12)
greenled.direction = digitalio.Direction.OUTPUT

blueled = digitalio.DigitalInOut(board.D11)
blueled.direction = digitalio.Direction.OUTPUT




button = digitalio.DigitalInOut(board.D22)
button.direction = digitalio.Direction.INPUT

from adafruit_motorkit import MotorKit
kit = MotorKit(i2c=board.I2C())
from adafruit_servokit import ServoKit
servokit = ServoKit(channels=8)
################################----
from threading import Thread
#########################################+++++++ датчик BNO085
from math import atan2, sqrt, pi
from board import SCL, SDA
from busio import I2C
from adafruit_bno08x import (
    BNO_REPORT_STEP_COUNTER,
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_GEOMAGNETIC_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = I2C(SCL, SDA, frequency=800000 )
bno = BNO08X_I2C(i2c , address=0x4A)

bno.enable_feature(BNO_REPORT_STEP_COUNTER)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
bno.enable_feature(BNO_REPORT_GEOMAGNETIC_ROTATION_VECTOR)

start_heading = 0

def find_heading(dqw, dqx, dqy, dqz):
    norm = sqrt(dqw * dqw + dqx * dqx + dqy * dqy + dqz * dqz)
    dqw = dqw / norm
    dqx = dqx / norm
    dqy = dqy / norm
    dqz = dqz / norm

    ysqr = dqy * dqy

    t3 = +2.0 * (dqw * dqz + dqx * dqy)
    t4 = +1.0 - 2.0 * (ysqr + dqz * dqz)
    yaw_raw = atan2(t3, t4)
    yaw = yaw_raw * 180.0 / pi
    if yaw > 0:
        yaw = 360 - yaw
    else:
        yaw = abs(yaw)
    return ((360+yaw-start_heading)%360)      # heading in 360 clockwise

quat_i, quat_j, quat_k, quat_real = bno.quaternion
start_heading = find_heading(quat_real, quat_i, quat_j, quat_k)


class BNO085:
    global bno,i2c,img
    def __init__(self):
        
        (self.quat_i, self.quat_j, self.quat_k, self.quat_real)=bno.quaternion
        self.heading = find_heading(self.quat_real, self.quat_i, self.quat_j, self.quat_k)
        self.stopped=False
    def start(self):
        Thread(target=self.update,args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            (self.quat_i, self.quat_j, self.quat_k, self.quat_real)=bno.quaternion
            self.heading = float(find_heading(self.quat_real, self.quat_i, self.quat_j, self.quat_k))
            self.heading=round((self.heading+45)%360,2) #45
            #time.sleep(0.05)
            try:
                toscreen=str(str("BNO"))+':'+str(self.heading)#+':'+str(self.heading)
                font.OverlayText(img, 1280, 720, toscreen, 280, 0, font.Blue, font.White)
            except:
                pass
    def read(self):
        return self.heading
    def stop(self):
        self.stopped=True


font = jetson.utils.cudaFont(size=16) # Шрифт для вывода на экран через jetson.utils

################################----Считываем камеру с нужными параметрами
camera = jetson.utils.videoSource("csi://0", argv=['--input-flip=rotate-0', '--input-width=640', '--input-height=480', '--input-rate=60'])
#camera = jetson.utils.videoSource("csi://0", argv=['--input-flip=rotate-0', '--input-width=960', '--input-height=540', '--input-frameRate=59'])
#camera = jetson.utils.videoSource("csi://0", argv=['--input-flip=rotate-0', '--input-width=1280', '--input-height=720', '--input-frameRate=59'])


################################----Задаём адрес стрима
#output=jetson.utils.videoOutput("rtp://192.168.8.104:5800",argv=['--headless']) # роутер Negotino2
output=jetson.utils.videoOutput("rtp://192.168.238.100:5800",argv=['--headless']) # роутер MTS (TP-LINK) стрим
#output=jetson.utils.videoOutput("rtp://192.168.1.104:5800",argv=['--headless']) # роутер Megafon
#output=jetson.utils.videoOutput("rtp://10.42.0.1:5800",argv=['--headless']) # точка доступа Jetson

################################----Вывод на дисплей----Запись на диск
#display = jetson.utils.videoOutput("display://0") # ' раскомментировать для вывода на дисплей
display = jetson.utils.videoOutput("video14.mp4") # раскомменировать для  записи на диск

################################----


flag_draw=True
x1, y1 = 60, 250
x2, y2 = 640-60, 400

Green=([47, 81, 74], [95, 255, 255])
Red_up=[[0, 179, 124], [9, 240, 255]]
Red_down=[[150,179,124], [179,240,255]]
Orange=([0, 50, 80], [50, 256, 256])
Blue=([99, 117, 73], [135, 255, 255])
Black=([0, 0, 0], [180, 255, 89])
###########################################################################################################

def make_mask1(frame, color):
    # функция создания маски цвета
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color[0]), np.array(color[1]))
    # print("make mask",color)
    return mask

############################################################################################################

def Find_box(frame, flag_draw=True):
    
    y_old=0
    color = Green
    x_green_banka = None # обнуление х координаты зеленого знака
    y_green_banka = None # обнуление у координаты зеленого знака
    area_green_banka = None # обнуление площади зеленого знака
    frame1=frame
    
    #Mask=make_mask_cuda(frame1, color)
    Mask=make_mask1(frame1, color)
    
    #contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # на отфильтрованной маске выделяем контуры
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # на отфильтрованной маске выделяем контуры
    
    
    for contour in contours: # перебираем все найденные контуры
        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура
        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        #print(area)
        if area > 900 and y>100: 
            #print(area)
            if y_old<y+h:
                y_old=y+h
                if x+w/2>275:
                    #x_green_banka = int(x+(abs(275-(x+w/2))/225)*(150-(y+h))/2) #R #####EGOR


                    #x_green_banka = int(x+w - (abs(275-(x+w/2))/225)*(150-(y+h))/2) #R
                    x_green_banka = int(x+(w/2))
                    #x_green_banka = int (abs((x+w)/2))
                    #x_green_banka = int (abs((x+w/2))/1)
                else:
                    #x_green_banka = int(x+(abs(275-(x+w/2))/225)*(150-(y+h))/2) #R ######EGOR

                    x_green_banka = int(x+(w/2))
                    #x_green_banka = int(x+w - (abs(275-(x+w/2))/225)*(150-(y+h))/2) #R
                    #x_green_banka = x
                    #x_green_banka = int (abs((x+w))/2)
                    #x_green_banka = int (abs((x+w/2))/1)
                #y_green_banka = y + h #R
                y_green_banka = int (abs((y + h/2))/1)
                area_green_banka = area
                if flag_draw:
                    c = (0, 0, 255)
                    if color == "green":
                        c = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
                    cv2.putText(frame, str(round(area, 1)), (x + x1, y - 20 + y1),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, c, 2)
                    cv2.putText(frame, str(x_green_banka)+ " " +str(y_green_banka), (x + x1, y - 40 + y1),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                    frame=cv2.line(frame,(x_green_banka-10,y_green_banka),(x_green_banka+10,y_green_banka),(0,0,0),2)
                    frame=cv2.line(frame,(x_green_banka,y_green_banka-10),(x_green_banka,y_green_banka+10),(0,0,0),2)

                    ################+++++++ рисуем прямоугольник на кубике
                    jetson.utils.cudaDrawLine(img, (x,y), (x+w,y), (0,0,0,200), 1) # верх горизонтальн
                    jetson.utils.cudaDrawLine(img, (x,y+h), (x+w,y+h), (0,0,0,200), 1) # низ горизонтальн
                    jetson.utils.cudaDrawLine(img, (x,y), (x,y+h), (0,0,0,200), 1) # лево вертикаль
                    jetson.utils.cudaDrawLine(img, (x+w,y), (x+w,y+h), (0,0,0,200), 1) # право вертикаль
                    ################-------


                    ################################################################################################
                    #jetson.utils.cudaDrawRect(img, (0,0,400,300), (255,127,0,200))
                    #jetson.utils.cudaDrawRect(img, (x,y,x+w,y+h), (255,127,0,200)) # рисуем прямоугольник на кубике
                    jetson.utils.cudaDrawLine(img, (x_green_banka-10,y_green_banka), (x_green_banka+10,y_green_banka), (0,0,0,200), 1)  # рисуем центр горизонт.
                    jetson.utils.cudaDrawLine(img, (x_green_banka,y_green_banka-10), (x_green_banka,y_green_banka+10), (0,0,0,200), 1) # рисуем центр веритик.
                    #################################################################################################
                    screen=str(str(int(x_green_banka)))+':'+str(y_green_banka)+':'+str(area_green_banka)
                    font.OverlayText(img, 1280, 720, screen, 15, 0, font.White, font.Blue)
    
    color = Red_up  # установление цвета знака который нужно найти
    x_red_banka = None # обнуление х координаты красного знака
    y_red_banka = None # обнуление у координаты красного знака
    area_red_banka = None # обнуление площади красного знака
    
    #Mask=make_mask_cuda(frame1, color)
    Mask=make_mask1(frame1, color)
    Mask2=make_mask1(frame1, Red_down)
    Mask=cv2.add(Mask,Mask2)
    #contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # на отфильтрованной маске выделяем контуры
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # на отфильтрованной маске выделяем контуры
    
    for contour in contours: # перебираем все найденные контуры
        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура
        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        #print(area)
        if area > 900:
            #print(area)
            if y_old<y+h:
                y_old=y+h
                if x+w/2>275:
                    #x_red_banka = int(x+w - (abs(275-(x+w/2))/225)*(150-(y+h))/2) #R

                    #x_red_banka = int(x+(abs(275-(x+w/2))/225)*(150-(y+h))/2) #R
                    x_red_banka = int(x+(w/2))
                    #x_red_banka = int (abs((x+w)/2))
                    #x_red_banka = int (abs((x+w/2))/1)
                else:
                    #x_red_banka = int(x+w - (abs(275-(x+w/2))/225)*(150-(y+h))/2) #R
                    #x_red_banka = int(x+(abs(275-(x+w/2))/225)*(150-(y+h))/2) #R
                    #x_red_banka = x+(w/2)
                    #x_green_banka = x
                    x_red_banka= int(x+(w/2))
                    #x_red_banka = int (abs((x+w))/2)
                    #x_red_banka = int (abs((x+w/2))/1)
                #y_green_banka = y + h #R
                y_red_banka = int (abs((y + h/2))/1)
                area_red_banka = area
                if flag_draw:
                    c = (0, 0, 255)
                    if color == "Red_up ":
                        c = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
                    cv2.putText(frame, str(round(area, 1)), (x + x1, y - 20 + y1),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, c, 2)
                    cv2.putText(frame, str(x_red_banka)+ " " +str(y_red_banka), (x + x1, y - 40 + y1),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                    frame=cv2.line(frame,(x_red_banka-10,y_red_banka),(x_red_banka+10,y_red_banka),(0,0,0),2)
                    frame=cv2.line(frame,(x_red_banka,y_red_banka-10),(x_red_banka,y_red_banka+10),(0,0,0),2)

                    ################+++++++ рисуем прямоугольник на кубике
                    jetson.utils.cudaDrawLine(img, (x,y), (x+w,y), (0,0,0,200), 1) # верх горизонтальн
                    jetson.utils.cudaDrawLine(img, (x,y+h), (x+w,y+h), (0,0,0,200), 1) # низ горизонтальн
                    jetson.utils.cudaDrawLine(img, (x,y), (x,y+h), (0,0,0,200), 1) # лево вертикаль
                    jetson.utils.cudaDrawLine(img, (x+w,y), (x+w,y+h), (0,0,0,200), 1) # право вертикаль
                    ################-------


                    ################################################################################################
                    #jetson.utils.cudaDrawRect(img, (0,0,400,300), (255,127,0,200))
                    #jetson.utils.cudaDrawRect(img, (x,y,x+w,y+h), (255,127,0,200))
                    jetson.utils.cudaDrawLine(img, (x_red_banka-10,y_red_banka), (x_red_banka+10,y_red_banka), (0,0,0,200), 1)
                    jetson.utils.cudaDrawLine(img, (x_red_banka,y_red_banka-10), (x_red_banka,y_red_banka+10), (0,0,0,200), 1)
                    #################################################################################################
                    screen=str(str(int(x_red_banka)))+':'+str(y_red_banka)+':'+str(area_red_banka)
                    font.OverlayText(img, 1280, 720, screen, 140, 0, font.White, font.Green)
             
    #return x_green_banka, y_green_banka, area_green_banka  # возвращение значений
    return x_red_banka, y_red_banka, area_red_banka, x_green_banka, y_green_banka, area_green_banka  # возвращение значений

def Find_start_line(frame_show, color):
    # функция поиска линии поворота

    x1, y1 = 320 - 40, 300
    x2, y2 = 320 + 40, 340

    frame_roi = frame_show[y1:y2, x1:x2] # вырезаем часть изображение
    cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 255), 2) # рисуем прямоугольник на изображении
    jetson.utils.cudaDrawRect(img, (x1,y1,x2,y2), (255,127,0,200))
    mask = make_mask1(frame_roi, color) # применяем маску для проверки
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # на отфильтрованной маске выделяем контуры
    for contour in contours: # перебираем все найденные контуры

        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура

        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        screen=str(str("Blue Area"))+':'+str(area)#+':'+str(self.heading)
        font.OverlayText(img, 1280, 720, screen, 25, 165, font.White, font.Green)
        if area > 0:
            cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
            return True
    return False

def Find_black_line_left(frame_show):
    # функция поиска черного бортика слева

  
    d=0

    x1, y1 = 0, 180-d
    x2, y2 = 20, 330

    frame_roi = frame_show[y1:y2, x1:x2]# вырезаем часть изображение

    cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)# рисуем прямоугольник на изображении
    jetson.utils.cudaDrawRect(img, (x1,y1,x2,y2), (255,127,0,200))
    mask = make_mask1(frame_roi, Black) # применяем маску для проверки
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # на отфильтрованной маске выделяем контуры

    flag_line = False
    max_y_left = 0
    for contour in contours: # перебираем все найденные контуры

        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура

        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        if area > 400:
            if flag_draw: # отрисовываем найденный контур прямоугольником
                cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (0, 0, 255), 2)
                jetson.utils.cudaDrawRect(img, (x,y1+y,x+w,y1+y+h), (255,0,0,100))
            if max_y_left < y + h:
                max_y_left = y + h

    cv2.putText(frame_show, "" + str(max_y_left), (0, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (0, 255, 0), 2)
    screen="Left wall"+":"+str(max_y_left)
    font.OverlayText(img, 1280, 720, screen, 15, 25, font.White, font.Blue)
    return max_y_left-d

def clamp(n, minn, maxn):  
    return max(min(maxn, n), minn)

def bnoreg(kp,kd):
    global direction,lasterror,servocenter,clamplow,clamphigh
    if abs(angle.read()-direction)>90:
        err=direction-angle.read()
    else:
        err=angle.read()-direction
    Upr = kp*err+kd*(err-lasterror)
    lasterror = err
    servoangle = servocenter-Upr
    servoangle = clamp(servoangle, clamplow, clamphigh)  
    servokit.servo[0].angle = servoangle 

sch=0
t=time.perf_counter()
#while display.IsStreaming():

angle=BNO085().start()
servocenter=74
clamplow=0
clamphigh=160
servokit.servo[0].angle = servocenter
state='find cub'
direction=45 #45  в 99 строке такое же значение
counter=0
lasterror=0
clock=-1
flag_alfa=0
timer1=0
timer2=0
delay=0.4

while True:
    
    img = camera.Capture() # кадр jetson.utils rgb8
    
    ###############################################################################
    bgr_img=jetson.utils.cudaAllocMapped(width=img.width,height=img.height,
    format='bgr8')
    jetson.utils.cudaConvertColor(img,bgr_img)
    #print (bgr_img)
    jetson.utils.cudaDeviceSynchronize()
    cv_img=jetson.utils.cudaToNumpy(bgr_img) # кадр OpenCV
    ################################################################################

    #getangle=angle.read()
    #print(getangle)

    
    #mask=make_mask1(cv_img, Green)
    #mask=make_mask1(cv_img, Red_up)

    frame=cv_img

    #x_green_banka, y_green_banka, area_green_banka= Find_box(frame) # получение данных о дорожных знаках
    x_red_banka, y_red_banka, area_red_banka, x_green_banka, y_green_banka, area_green_banka = Find_box(frame) # возвращение значений
    #Find_black_line_left(frame)

    font.OverlayText(img, 1280, 720, state, 400, 30, font.Blue, font.White)
    font.OverlayText(img, 1280, 720, str(delay), 400, 60, font.Blue, font.White)

    output.Render(img)  # раскомментировать для стрима
    #display.Render(img) # раскомментировать для записи или для вывода на дисплей
    if button.value==False:#cv2.waitKey(1)==ord('q'): ##### счетчик
       break
    
kit.motor1.throttle = 0
angle.stop()
servokit.servo[0].angle = servocenter
