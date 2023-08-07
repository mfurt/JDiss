import numpy as np
from numpy import pi

from consts import EMRAT, AU


def reduce(angle):
    while (angle > 2 * pi):
        angle -= 2 * pi
    while (angle <= 0):
        angle += 2 * pi
    return angle

def ephem(JD, target):
    # * Получение координат и скорстей в удобоваримом виде и единицах
    # * Пересчёт координат относительно Солнца, а не барицентра СС
    # * Вычисление координат Земли, а не барицентра ЗЛ
    qv = eph.position_and_velocity('sun', JD)
    qsun = qv[0]
    vsun = qv[1]

    target = target.lower() # Чтобы не промахнуться с регистром

    if (target == 'earth'): # Достаём координаты Земли из барицентра
        qv = eph.position_and_velocity('moon', JD)
        qmoon = qv[0]
        vmoon = qv[1]

        qv = eph.position_and_velocity('earthmoon', JD)
        q = qv[0] - (1./(1 + EMRAT))*qmoon
        v = qv[1] - (1./(1 + EMRAT))*vmoon

    else:
        qv = eph.position_and_velocity(target, JD)
        q = qv[0]
        v = qv[1]

    if (target != 'earth'):  # Луна в геоцентрической системе
        q -= qsun
        v -= vsun

    # v /= 86400 # Приводим к км/с

    # Магия работы с эфемеридами:
    q = q.transpose()[0]
    v = v.transpose()[0]
    return np.array([q[0], q[1], q[2], v[0], v[1], v[2]])/AU
