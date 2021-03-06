# импортируем необходимые библиотеки для вычислений
import time
import numpy as np
import jetson.inference
import jetson.utils
import cv2
import board
import digitalio
import smbus
from threading import Thread

# задаем необхыдимы константы и классы для измерения заряда батареи

# Config Register (R/W)
_REG_CONFIG                 = 0x00
# SHUNT VOLTAGE REGISTER (R)
_REG_SHUNTVOLTAGE           = 0x01

# BUS VOLTAGE REGISTER (R)
_REG_BUSVOLTAGE             = 0x02

# POWER REGISTER (R)
_REG_POWER                  = 0x03

# CURRENT REGISTER (R)
_REG_CURRENT                = 0x04

# CALIBRATION REGISTER (R/W)
_REG_CALIBRATION            = 0x05

class BusVoltageRange:
    """Constants for ``bus_voltage_range``"""
    RANGE_16V               = 0x00      # set bus voltage range to 16V
    RANGE_32V               = 0x01      # set bus voltage range to 32V (default)

class Gain:
    """Constants for ``gain``"""
    DIV_1_40MV              = 0x00      # shunt prog. gain set to  1, 40 mV range
    DIV_2_80MV              = 0x01      # shunt prog. gain set to /2, 80 mV range
    DIV_4_160MV             = 0x02      # shunt prog. gain set to /4, 160 mV range
    DIV_8_320MV             = 0x03      # shunt prog. gain set to /8, 320 mV range

class ADCResolution:
    """Constants for ``bus_adc_resolution`` or ``shunt_adc_resolution``"""
    ADCRES_9BIT_1S          = 0x00      #  9bit,   1 sample,     84us
    ADCRES_10BIT_1S         = 0x01      # 10bit,   1 sample,    148us
    ADCRES_11BIT_1S         = 0x02      # 11 bit,  1 sample,    276us
    ADCRES_12BIT_1S         = 0x03      # 12 bit,  1 sample,    532us
    ADCRES_12BIT_2S         = 0x09      # 12 bit,  2 samples,  1.06ms
    ADCRES_12BIT_4S         = 0x0A      # 12 bit,  4 samples,  2.13ms
    ADCRES_12BIT_8S         = 0x0B      # 12bit,   8 samples,  4.26ms
    ADCRES_12BIT_16S        = 0x0C      # 12bit,  16 samples,  8.51ms
    ADCRES_12BIT_32S        = 0x0D      # 12bit,  32 samples, 17.02ms
    ADCRES_12BIT_64S        = 0x0E      # 12bit,  64 samples, 34.05ms
    ADCRES_12BIT_128S       = 0x0F      # 12bit, 128 samples, 68.10ms

class Mode:
    """Constants for ``mode``"""
    POWERDOW                = 0x00      # power down
    SVOLT_TRIGGERED         = 0x01      # shunt voltage triggered
    BVOLT_TRIGGERED         = 0x02      # bus voltage triggered
    SANDBVOLT_TRIGGERED     = 0x03      # shunt and bus voltage triggered
    ADCOFF                  = 0x04      # ADC off
    SVOLT_CONTINUOUS        = 0x05      # shunt voltage continuous
    BVOLT_CONTINUOUS        = 0x06      # bus voltage continuous
    SANDBVOLT_CONTINUOUS    = 0x07      # shunt and bus voltage continuous


class INA219:
    def __init__(self, i2c_bus=1, addr=0x40):
        self.bus = smbus.SMBus(i2c_bus);
        self.addr = addr

        # Set chip to known config values to start
        self._cal_value = 0
        self._current_lsb = 0
        self._power_lsb = 0
        self.set_calibration_32V_2A()

    def read(self,address):
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return ((data[0] * 256 ) + data[1])

    def write(self,address,data):
        temp = [0,0]
        temp[1] = data & 0xFF
        temp[0] =(data & 0xFF00) >> 8
        self.bus.write_i2c_block_data(self.addr,address,temp)

    def set_calibration_32V_2A(self):
        """Configures to INA219 to be able to measure up to 32V and 2A of current. Counter
           overflow occurs at 3.2A.
           ..note :: These calculations assume a 0.1 shunt ohm resistor is present
        """
        # By default we use a pretty huge range for the input voltage,
        # which probably isn't the most appropriate choice for system
        # that don't use a lot of power.  But all of the calculations
        # are shown below if you want to change the settings.  You will
        # also need to change any relevant register settings, such as
        # setting the VBUS_MAX to 16V instead of 32V, etc.

        # VBUS_MAX = 32V             (Assumes 32V, can also be set to 16V)
        # VSHUNT_MAX = 0.32          (Assumes Gain 8, 320mV, can also be 0.16, 0.08, 0.04)
        # RSHUNT = 0.1               (Resistor value in ohms)

        # 1. Determine max possible current
        # MaxPossible_I = VSHUNT_MAX / RSHUNT
        # MaxPossible_I = 3.2A

        # 2. Determine max expected current
        # MaxExpected_I = 2.0A

        # 3. Calculate possible range of LSBs (Min = 15-bit, Max = 12-bit)
        # MinimumLSB = MaxExpected_I/32767
        # MinimumLSB = 0.000061              (61uA per bit)
        # MaximumLSB = MaxExpected_I/4096
        # MaximumLSB = 0,000488              (488uA per bit)

        # 4. Choose an LSB between the min and max values
        #    (Preferrably a roundish number close to MinLSB)
        # CurrentLSB = 0.0001 (100uA per bit)
        self._current_lsb = .1  # Current LSB = 100uA per bit

        # 5. Compute the calibration register
        # Cal = trunc (0.04096 / (Current_LSB * RSHUNT))
        # Cal = 4096 (0x1000)

        self._cal_value = 4096

        # 6. Calculate the power LSB
        # PowerLSB = 20 * CurrentLSB
        # PowerLSB = 0.002 (2mW per bit)
        self._power_lsb = .002  # Power LSB = 2mW per bit

        self.write(_REG_CALIBRATION,self._cal_value)

        self.bus_voltage_range = BusVoltageRange.RANGE_32V
        self.gain = Gain.DIV_8_320MV
        self.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.mode = Mode.SANDBVOLT_CONTINUOUS
        self.config = self.bus_voltage_range << 13 | \
                      self.gain << 11 | \
                      self.bus_adc_resolution << 7 | \
                      self.shunt_adc_resolution << 3 | \
                      self.mode
        self.write(_REG_CONFIG,self.config)

    def getShuntVoltage_mV(self):
        self.write(_REG_CALIBRATION,self._cal_value)
        value = self.read(_REG_SHUNTVOLTAGE)
        if value > 32767:
            value -= 65535
        return value * 0.01

    def getBusVoltage_V(self):
        self.write(_REG_CALIBRATION,self._cal_value)
        self.read(_REG_BUSVOLTAGE)
        return (self.read(_REG_BUSVOLTAGE) >> 3) * 0.004

    def getCurrent_mA(self):
        value = self.read(_REG_CURRENT)
        if value > 32767:
            value -= 65535
        return value * self._current_lsb

    def getPower_W(self):
        self.write(_REG_CALIBRATION,self._cal_value)
        value = self.read(_REG_POWER)
        if value > 32767:
            value -= 65535
        return value * self._power_lsb
        
ina219 = INA219(addr=0x42)

# инициализируем светодиоды и кнопку на верхней панели

redcubled = digitalio.DigitalInOut(board.D26) 
greencubled = digitalio.DigitalInOut(board.D20) 


redcubled.direction = digitalio.Direction.OUTPUT
greencubled.direction = digitalio.Direction.OUTPUT

redled = digitalio.DigitalInOut(board.D13)
redled.direction = digitalio.Direction.OUTPUT

whiteled = digitalio.DigitalInOut(board.D12)
whiteled.direction = digitalio.Direction.OUTPUT

greenled = digitalio.DigitalInOut(board.D12)
greenled.direction = digitalio.Direction.OUTPUT

blueled = digitalio.DigitalInOut(board.D11)
blueled.direction = digitalio.Direction.OUTPUT

button = digitalio.DigitalInOut(board.D22)
button.direction = digitalio.Direction.INPUT

# импортируем необходимые библиотеки для управления мотором и сервомотором

from adafruit_motorkit import MotorKit
kit = MotorKit(i2c=board.I2C())
from adafruit_servokit import ServoKit
servokit = ServoKit(channels=8)

# импортируем модули для BNO085

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

# функция поиска угла

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

# class для взаимодействия с BNO085

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
            self.heading=int((self.heading+45)%360) #45
    def read(self):
        return self.heading
    def stop(self):
        self.stopped=True

# инициализируем камеры и доп. модули

camera = jetson.utils.videoSource("csi://0", argv=['--input-flip=rotate-0', '--input-width=640', '--input-height=480', '--input-rate=60'])
camera_line = jetson.utils.videoSource("csi://1", argv=['--input-flip=rotate-0', '--input-width=640', '--input-height=480', '--input-rate=60'])
output=jetson.utils.videoOutput("rtp://192.168.108.102:5800",argv=['--headless']) # роутер MTS (TP-LINK) стрим
display = jetson.utils.videoOutput("video14.mp4",argv=['--headless']) # раскомменировать для  записи на диск


# функция создания маски цвета

def create_mask(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color[0]), np.array(color[1]))
    # print("make mask",color)
    return mask

# функция поиска кубика на экране

def find_cub(frame):

    global area_porog,area_max
    y_prev=0
    color = Green
    x_green_cub = None
    y_green_cub = None 
    area_green_cub = None 
    h_green_cub=0

    
    Mask=create_mask(frame, color)
    
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # на отфильтрованной маске выделяем контуры
    
    for contour in contours: # перебираем все найденные контуры
        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура
        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        #print(area)
        if area > area_porog and y>50: 
            if y_prev<y+h:
                y_prev=y+h
                h_green_cub=h
                x_green_cub = int(x + w/2)    
                y_green_cub = int(y + h/2)
                area_green_cub = area
                area_max=max(area_max,area)
                if area>area_porog: area_porog=area
                cv2.rectangle(frame, (x, y), (x + w, y + h), green_cub_color, 2)
                cv2.putText(frame, str(round(area, 1)), (x + w, y+h+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, cub_text_color, 2)
                cv2.putText(frame, str(x_green_cub)+ " " +str(y_green_cub), (x + w, y+h+25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, cub_text_color, 2)
                frame=cv2.line(frame,(x_green_cub-10,y_green_cub),(x_green_cub+10,y_green_cub),(0,0,0),2)
                frame=cv2.line(frame,(x_green_cub,y_green_cub-10),(x_green_cub,y_green_cub+10),(0,0,0),2)
                frame=cv2.line(frame,(320,480),(x_green_cub,y_green_cub),green_cub_color,2)

    color = Red_up  
    x_red_cub = None 
    y_red_cub = None 
    area_red_cub = None 
    h_red_cub=0
    Mask=create_mask(frame, color)
    Mask2=create_mask(frame, Red_down)
    Mask=cv2.add(Mask,Mask2)
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # на отфильтрованной маске выделяем контуры
    
    for contour in contours: # перебираем все найденные контуры
        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура
        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        if area > area_porog and y>50:
            if y_prev<y+h:
                y_prev=y+h
                h_red_cub=h
                x_red_cub = int(x + w/2)
                y_red_cub = int(y + h/2)
                area_red_cub = area
                if area>area_porog: area_porog=area
                area_max=max(area_max,area)
                cv2.rectangle(frame, (x, y), (x + w, y + h), red_cub_color, 2)
                cv2.putText(frame, str(round(area, 1)), (x + w, y+h+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, cub_text_color, 2)
                cv2.putText(frame, str(x_red_cub)+ " " +str(y_red_cub), (x + w, y+h+25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, cub_text_color, 2)
                frame=cv2.line(frame,(x_red_cub-10,y_red_cub),(x_red_cub+10,y_red_cub),(0,0,0),2)
                frame=cv2.line(frame,(x_red_cub,y_red_cub-10),(x_red_cub,y_red_cub+10),(0,0,0),2)
                frame=cv2.line(frame,(320,480),(x_red_cub,y_red_cub+10),red_cub_color,2)
                   
             
 
    return x_red_cub, y_red_cub, area_red_cub, h_red_cub, x_green_cub, y_green_cub, area_green_cub, h_green_cub  

# функция поиска стартовой линии

def find_line(color):
    x1, y1 = 320 - 150, 0
    x2, y2 = 320 + 150, 100
    frame_roi = frame_line[y1:y2, x1:x2] # вырезаем часть изображение
    cv2.rectangle(frame_line, (x1, y1), (x2, y2), (0, 255, 255), 2) # рисуем прямоугольник на изображении
    mask = create_mask(frame_roi, color) # применяем маску для проверки
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # на отфильтрованной маске выделяем контуры
    for contour in contours: # перебираем все найденные контуры
        x, y, w, h = cv2.boundingRect(contour) # Создаем прямоугольник вокруг контура
        area = cv2.contourArea(contour) # вычисляем площадь найденного контура
        if area > 50:
            cv2.rectangle(frame_line, (x1+x, y1+y), (x1+x+w, y1+y+h), (255, 0, 0), 2)
            return True
    return False

# функция ограничения диапазонов

def clamp(n, minn, maxn):  
    return max(min(maxn, n), minn)

# регулятор для движения согласну курсу BNO085

def bnoreg(kp,kd):
    global direction,lasterror,servocenter,clamplow,clamphigh,clock
    raw_error=angle.read()-direction+(0.25*counter*clock)
    if abs(raw_error)<180: err=raw_error
    else:
        if raw_error<0: err=raw_error%180
        else: err=raw_error%180-180

    Upr = kp*err+kd*(err-lasterror)
    lasterror = err
    servoangle = servocenter-Upr
    servoangle = clamp(servoangle, clamplow, clamphigh)  
    servokit.servo[0].angle = servoangle  

# функция нахождения координат вектора

def make_vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]

# функции преобразования cuda img в opencv img, convertcolor, и обратно

def cuda_to_cv(source):
    global img
    img = source.Capture()
    bgr_img=jetson.utils.cudaAllocMapped(width=img.width,height=img.height,
    format='bgr8')
    jetson.utils.cudaConvertColor(img,bgr_img)
    #print (bgr_img)
    jetson.utils.cudaDeviceSynchronize()
    cv_img=jetson.utils.cudaToNumpy(bgr_img)
    return cv_img
def cv_to_cuda(source):
    #global frame 
    bgr_img = jetson.utils.cudaFromNumpy(source, isBGR=True)
    rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,height=bgr_img.height,format='rgb8')
    jetson.utils.cudaConvertColor(bgr_img, rgb_img)
    return rgb_img

# функции вывода заряда батареи и телеметрии на экран

def put_battery():
    global y_0
    bus_voltage = ina219.getBusVoltage_V()             # voltage on V- (load side)
    shunt_voltage = ina219.getShuntVoltage_mV() / 1000 # voltage between V+ and V- across the shunt
    current = ina219.getCurrent_mA()                   # current in mA
    power = ina219.getPower_W()                        # power in W
    p = (bus_voltage - 6)/2.4*100
    if(p > 100):p = 100
    if(p < 0):p = 0

    cv2.putText(frame,"Percent: "+str(round(p,2))+"%"  ,(550, y_max-10),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
    cv2.putText(frame,"Power: "+str(round(power,2))+"W"  ,(550, y_max-20),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
    cv2.putText(frame,"Current: "+str(round(current/1000,2))+"A"  ,(550, y_max-30),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
    cv2.putText(frame,"Load Voltage: "+str(round(bus_voltage,2))+"V"  ,(550, y_max-40),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
    #cv2.putText(frame,"Shunt Voltage: "+str(round(shunt_voltage,2))+"V"  ,(550, y_max-50),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
    #cv2.putText(frame,"PSU Voltage: "+str(round(bus_voltage + shunt_voltage,2))+"V"  ,(550, y_max-60),cv2.FONT_HERSHEY_SIMPLEX, 0.25, green_cub_color, 1)
def put_telemetry():
    global x_red_cub, y_red_cub, area_red_cub, x_green_cub, y_green_cub, area_green_cub, angle_cub,direction,state,servoangle,h_red_cub,h_green_cub,angle_cub_calc,timer_cub,area_porog,clamplow, timer_mid, k_h, h_true_green, h_true_red
    cv2.putText(frame,'r_a: '+str(area_red_cub)+' x_red: '+str(x_red_cub)+' y_red: '+str(y_red_cub)+' h_red: '+str(h_red_cub),(0, y_max-470),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'g_a: '+str(area_green_cub)+' x_gr: '+str(x_green_cub)+' y_gr: '+str(y_green_cub)+' h_green: '+str(h_green_cub),(0, y_max-450),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'d_angle: '+str(angle_cub)+'/'+str(angle_cub_calc_red)+'/'+str(angle_cub_calc_green),(0, y_max-430),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'BNO: '+str(angle.read()),(0, y_max-410),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'direction: '+str(direction),(0, y_max-390),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'state: '+str(state),(0, y_max-370),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'servo: '+str(time.perf_counter()-timer_mid),(0, y_max-350),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'tm: '+str(t_manevra),(0, y_max-330),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)
    cv2.putText(frame,'htrue: '+str(h_true_green)+'/'+str(h_true_red),(0, y_max-310),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_cub_color, 2)

# функия pd регулятора по x координате кубика

def pd(err):
    global kp,kd,lasterror,servocenter,servoangle,clamphigh,clamplow
    Upr = kp*err+kd*(err-lasterror)
    lasterror = err
    servoangle = servocenter+Upr+Upr*kp
    servoangle = clamp(servoangle, clamplow, clamphigh)  
    servokit.servo[0].angle = servoangle 

# функция рассчета угла отклонения кубика

def calc_cub_angle():
    global angle_cub, direction, x_red_cub, x_green_cub, mn, angle_cub_calc_red,angle_cub_calc_green, mn, clock
    angle_cub_calc_red=0
    angle_cub_calc_green=0
    if area_green_cub:
        a=[320,480]
        b=[x_green_cub,y_green_cub]
        c=[320,y_green_cub]
        vector_1 = make_vector(a, b)
        vector_2 = make_vector(a, c)
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle_cub_calc_green = int(np.arccos(dot_product)*(180/np.pi))
        if clock==1:
            if x_green_cub>=320: mn=0.8
            else: mn=-0.9
            angle_cub_calc_green=(angle_cub_calc_green-angle.read()+direction)*mn
        if clock==-1:
            if x_green_cub>=320: mn=1
            else: mn=-1   
            angle_cub_calc_green=(angle_cub_calc_green+angle.read()-direction)*mn
    if area_red_cub:
        a=[320,480]
        b=[x_red_cub,y_red_cub]
        c=[320,y_red_cub]
        vector_1 = make_vector(a, b)
        vector_2 = make_vector(a, c)
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle_cub_calc_red = int(np.arccos(dot_product)*(180/np.pi))
        if clock==1:
            if x_red_cub>=320: mn=-0.95
            else: mn=1   
            angle_cub_calc_red=(angle_cub_calc_red-angle.read()+direction)*mn
        if clock==-1:
            if x_red_cub>=320: mn=-1
            else: mn=1.15
            angle_cub_calc_red=(angle_cub_calc_red+angle.read()-direction)*mn

# функция подсчета кубиков     

def calccub():
    global counter,redcntr,greencntr
    if counter<=4:
        if area_green_cub is not None:greencntr+=1
        if area_red_cub is not None:redcntr+=1


# задаем значения необходимых переменных


red_cub_color=(0, 0, 255)
green_cub_color=(0,255,0)
cub_text_color=(0,255,0)#(250,50,255)
info_text_color=(255,255,0)
y_max=480


Green=([47, 81, 74], [95, 255, 255])
Red_up=[[0, 179, 100], [9, 240, 255]]
Red_down=[[150,179,100], [179,240,255]]
Orange=([0, 105, 106], [25, 175, 166])
Blue=([99, 117, 73], [135, 255, 255])
Black=([0, 0, 0], [180, 255, 89])


sch=0
angle_cub=None
servocenter=80
clamplow=5
clamphigh=170
servoangle=80
servokit.servo[0].angle = servocenter
state=-1
direction=45 
counter=0
lasterror=0
clock=1
flag_alfa=0
timer_mid=0
delay=0.4
timer_cub=0
t_manevra=0
mn=1
angle_cub_calc=0
complete_red=False
area_porog=700
area_max=0
angle_cub_calc_red=0
angle_cub_calc_green=0
mn=0
cub_color=0
direction_prev=0
k_h=1
h_true_red=0
h_true_green=0
redcntr=0
greencntr=0

# сбрасываем таймеры,стартуем поток BNO

timer_start=time.perf_counter()
timer_for_finish=time.perf_counter()
t_finish=time.perf_counter()
angle=BNO085().start()


while True:
    # включаем камеры, ждем нажатия кнопки, когда кнопка нажата, делаем плавный разгон и начинаем проезд
    if time.perf_counter()-timer_start>4 and state==-1:
        blueled.value=True
        if button.value==False:
            blueled.value=False
            for i in range(50):
                kit.motor1.throttle = i/100
                time.sleep(0.01)
            kit.motor1.throttle = 0.5
            state='start'

    # считываем изображения с камер
    frame=cuda_to_cv(camera)
    frame_line=cuda_to_cv(camera_line)


    x_red_cub, y_red_cub, area_red_cub, h_red_cub, x_green_cub, y_green_cub, area_green_cub, h_green_cub = find_cub(frame) # получение значений


    if state=='start': # едем до линии, если видим линию
        if find_line(frame_line,Blue,'default'): clock=-1; state=0; counter+=1; finish_color=Blue; t_finish=2.7 # линия синяя, направление против часовой
        if find_line(frame_line,Orange,'default'): clock=1; state=0; counter+=1; finish_color=Orange; t_finish=4 # линяя оранжевая, направление по часовой


    if state==0: #начало поворота
        area_porog=700        
        whiteled.value=True
        state=1
        direction=(direction+70*clock+360)%360
        servokit.servo[0].angle = 5
        

    if state==1: # читаем кубики на повороте
        area_porog=800 
        area_max=0
        turntype=0
        complete_red=False
        mid_proezd=0
        # если видим кубик, задаем параметры пд регулятора
        if abs(direction-angle.read())<5 or area_red_cub is not None or area_green_cub is not None:
            whiteled.value=False
            state=2
            if area_red_cub is not None: # видим красный
                calccub()
                turntype=5
                redcubled.value=True 
                h_true_red=h_red_cub*math.cos(math.radians(angle.read()-direction))
                if h_red_cub>=85:h_true_red=max(h_true_red,h_red_cub)
                if h_true_red<65: state=6; hardred=False; treb=80; kp=0.4; kd=0.8 # кубик на середине
                else: 
                    kit.motor1.throttle = 0.45
                    state=2
                    timer_red=time.perf_counter()
                    if x_red_cub>70: 
                        servokit.servo[0].angle = servocenter
                        hardred=True
                        redled.value=True 
                        blueled.value=True 
                        while time.perf_counter()-timer_red<1.2:#1.9 
                            whiteled.value=True
                            pass
                    else: hardred=False; treb=80; kp=0.6; kd=0.8; clamplow=35

            if area_green_cub is not None: # видим зеленый
                calccub()
                greencubled.value=True
                turntype=3
                h_true_green=h_green_cub*math.cos(math.radians(angle.read()-direction))
                if h_green_cub>=85: h_true_green=max(h_true_green,h_green_cub)
                if h_true_green<65: state=6; hardgreen=False; treb=520; kp=0.3; kd=0.6; # кубик на середине
                else:
                    kit.motor1.throttle = 0.6
                    state=2
                    if x_green_cub>130: hardgreen=True; redled.value=True; blueled.value=True
                    else: hardgreen=False; treb=640; kp=0.4; kd=0.6; time.sleep(0.3) 
            
            direction=(direction+20*clock+360)%360   


    if state==2: # выполняем поворот
        if turntype==5: # поворот по красному кубику
            if area_red_cub is not None and hardred==False and complete_red==False:
                area_porog=700
                pd(x_red_cub-treb)  
                x_red_prev=x_red_cub  
            elif hardred==False : 
                area_porog=area_max
                pd(x_red_prev-treb)
            if area_red_cub is None: complete_red=True; bnoreg(2,4)
            if hardred==True:
                bnoreg(3,6)#3
            if abs(direction-angle.read())<5: 
                if h_red_cub>60 and complete_red==False: hardred=True
                else: state=3; timer_mid=time.perf_counter(); clamplow=5; mid_time=0.25
        if turntype==3: # поворот по зеленому кубику
            if area_green_cub is not None and hardgreen==False:
                pd(x_green_cub-treb) 
            if area_green_cub is None or hardgreen==True: 
                bnoreg(3,6)#3
                if abs(direction-angle.read())<10: state=3; timer_mid=time.perf_counter(); mid_time=0.35#counter+=1
        if turntype==0: # поворот без кубика
            bnoreg(2,4)
            if abs(direction-angle.read())<5:  
                if find_line(frame_line,Orange,'custom'):
                    mid_proezd=time.perf_counter()
                    if area_red_cub is not None: calccub(); turntype=5; redcubled.value=True; state=6; hardred=False; treb=80; kp=0.4; kd=0.8;  #mid_area_limit=750
                    elif area_green_cub is not None: calccub(); turntype=3; greencubled.value=True; state=6; hardgreen=False; treb=520; kp=0.3; kd=0.6
                    else: state=3; timer_mid=time.perf_counter(); mid_time=1.5
                else: state=3; timer_mid=time.perf_counter(); mid_time=0.5


    if state==3: # едем до дальних
        kit.motor1.throttle = 0.5
        bnoreg(3,6)
        area_porog=700
        if abs(direction-angle.read())<5:
            calc_cub_angle()
            red_ready=False
            green_ready=False
            # проверяем кубики на прямой
            if area_red_cub is not None and y_red_cub<200: # видим красный
                calccub()
                cub_color=5
                direction_prev=direction
                delta_direction=(angle_cub_calc_red-10)*1.85#1.5
                direction=(direction-delta_direction*(-clock)+360)%360 #+==>-
                t_manevra=0.7+0.125*abs(angle_cub_calc_red)-0.0023*abs(angle_cub_calc_red)**2 #magic do not touch
                k_h=1-0.025*(h_red_cub-45)
                k_h=clamp(k_h,0,1)
                t_manevra=t_manevra*k_h
                state=4
                timer_cub=time.perf_counter()
            if area_green_cub is not None and y_green_cub<200: # видим зеленый
                calccub()
                cub_color=3
                direction_prev=direction
                delta_direction=(angle_cub_calc_green-10)*1.8#1.8
                direction=(direction+delta_direction*(-clock)+360)%360
                t_manevra=0.7+0.125*abs(angle_cub_calc_green)-0.0023*abs(angle_cub_calc_green)**2 #magic do not touch
                k_h=1-0.025*(h_green_cub-45)
                k_h=clamp(k_h,0,1)
                t_manevra=t_manevra*k_h
                state=4
                timer_cub=time.perf_counter()
        if time.perf_counter()-timer_mid>mid_time and cub_color==0: # если кубиков нет, просто едем до линии
            state=5


    if state==4: # объезд + при отработке на далний возвращаемся на курс
        bnoreg(2,4)      
        if cub_color==5: 
            if area_red_cub is None or y_red_cub>190: red_ready=True
            if time.perf_counter()-timer_cub>abs(t_manevra) and red_ready==True : direction=direction_prev; state=5; area_porog=700#or abs(direction-angle.read())<2
        if cub_color==3:
            if area_green_cub is None or y_green_cub>190: green_ready=True
            if time.perf_counter()-timer_cub>abs(t_manevra) and green_ready==True : direction=direction_prev; state=5; area_porog=700#or abs(direction-angle.read())<2    


    if state==5: # едем до линии
        area_porog=700    
        kit.motor1.throttle = 0.5
        bnoreg(3,6)
        if abs(direction-angle.read())<5 and find_line(frame_line,finish_color,'default'): state=0; counter+=1; timer_for_finish=time.perf_counter()


    if state==6: # объезд кубиков на середине (вспомогательный state)
        kit.motor1.throttle = 0.5
        if time.perf_counter()-mid_proezd>0.8:
            if turntype==5: # объезд красного
                if area_red_cub is not None:
                    pd(x_red_cub-treb)
                if area_red_cub is None: 
                    bnoreg(2,4) 
                if (abs(direction-angle.read())<5 and area_red_cub is None) or (abs(direction-angle.read())<5 and area_red_cub<1500): 
                    state=5; timer_mid=time.perf_counter(); clamplow=5
            if turntype==3: # объезд зеленого
                if area_green_cub is not None:
                    pd(x_green_cub-treb)
                if area_green_cub is None: 
                    bnoreg(2,4)
                if abs(direction-angle.read())<5 and area_green_cub is None: 
                    state=5; timer_mid=time.perf_counter()
        else: pass   



    put_battery() # выводим заряд батареи
    put_telemetry() # выводим телеметрию
    display.Render(cv_to_cuda(frame)) # записываем на диск 
    if button.value==False and state!=-1: # если кнопка нажата, выключаем программу (экстренная ситуация)
       break
    if time.perf_counter()-timer_for_finish>t_finish and counter==12: # как только проехали 3 круга - останавливаемся
        break
kit.motor1.throttle = 0
angle.stop()
servokit.servo[0].angle = servocenter
print(redcntr,greencntr)