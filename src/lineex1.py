                                                            #
from time import sleep
from pyiArduinoI2Cbumper import *                           #   Подключаем библиотеку для работы с бампером I2C-flash.
bum = pyiArduinoI2Cbumper(0x09)                             #   Объявляем объект bum для работы с функциями и методами библиотеки pyiArduinoI2Cbumper, указывая адрес модуля на шине I2C.
                                                           #   Если объявить объект без указания адреса bum = pyiArduinoI2Cbumper(), то адрес будет найден автоматически.
while True:                                                 #
    t = "Аналоговые значения датчиков 1-9: "\
        "{}, {},".format(
            bum.getLineAnalog(1),                           #   Значение АЦП снятое с 2 датчика линии.
            # bum.getLineAnalog(2),                           #   Значение АЦП снятое с 2 датчика линии.
            # bum.getLineAnalog(3),                           #   Значение АЦП снятое с 3 датчика линии.
            # bum.getLineAnalog(4),                           #   Значение АЦП снятое с 4 датчика линии.
            # bum.getLineAnalog(5),                           #   Значение АЦП снятое с 5 датчика линии.
            # bum.getLineAnalog(6),                           #   Значение АЦП снятое с 6 датчика линии.
            # bum.getLineAnalog(7),                           #   Значение АЦП снятое с 7 датчика линии.
            # bum.getLineAnalog(8),                           #   Значение АЦП снятое с 8 датчика линии.
            bum.getLineAnalog(9)                            #   Значение АЦП снятое с 9 датчика линии.
            )
    print(t)
    sleep(.2)
    