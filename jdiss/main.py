import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from numpy import sqrt, dot, sin, cos, arctan2, cross, arcsin, random

from consts import *
from utils import reduce

np.random.seed(777)
# Все параметры записывают перед процедурками
interval = 5  # интервал
n2 = 12  # кол наблюдений
kk = 5.5694745387792770762  # квантиль фишера
kt = 1000  # кол точек
kd = 0  # дни

# объект
t0 = 2453993.38341912
a = 2.6326375423103763
e = 0.68296434449828680
i = 19.0332 * rad
W = 7.05843 * rad
w = 53.0262 * rad
M0 = 348.065 * rad

# земля
a2 = 1.00000267492667613
e2 = 0.01635277395291997
i2 = 0.40907645295092280
W2 = 0.00001077572125351
w2 = 1.805121824028950
M02 = -1.917372742782691


class Sat:
    def __init__(self, init=None, ptype='kep', t0=t0, mu=mu_geo):
        self.t0 = t0
        self.mu = mu

        if (ptype == 'kep'):
            if (init == None):
                init = [42164, 0, 0, 0, 0, 0]  # дефолт
            self.a, self.e, self.i, self.W, self.w, self.M0 = init
            self.cartesian()

        if (ptype == 'qv'):
            self.x, self.y, self.z, self.vx, self.vy, self.vz = init
            self.findOrbit()

    ########################## Вычисления ##########################

    ##### Методы определения декартовых координат #####
    def cartesian(self, t=t0, dt=None):
        a, e, i, W, w, M0 = self.get('kep')
        mu = self.mu
        t0 = self.t0

        A = [[cos(W), -sin(W), 0],
             [sin(W), cos(W), 0],
             [0, 0, 1]]
        A = np.matrix(A)

        B = [[1, 0, 0],
             [0, cos(i), -sin(i)],
             [0, sin(i), cos(i)]]
        B = np.matrix(B)

        C = [[cos(w), -sin(w), 0],
             [sin(w), cos(w), 0],
             [0, 0, 1]]
        C = np.matrix(C)

        R = A * B * C  # поворотная матрица

        n = sqrt(mu * a ** (-3))

        if (dt):
            M = M0 + n * dt
        else:
            M = M0 + n * (t - t0)

        M = reduce(M)

        # Численное решение уравнения Кеплера
        E = M
        while (abs(E - e * sin(E) - M) > 1e-12):
            E = E - (E - e * sin(E) - M) / (1 - e * cos(E))

        q = np.matrix([a * (cos(E) - e), a * sqrt(1 - e ** 2) * sin(E), 0])
        dq = np.matrix([-a * n * sin(E) / (1 - e * cos(E)), a * sqrt(1 - e ** 2) * cos(E) * n / (1 - e * cos(E)), 0])

        #        reduced = reduce(M) - M0

        q = R * q.T
        dq = R * dq.T

        self.x, self.y, self.z = q.A1
        self.vx, self.vy, self.vz = dq.A1
        return [self.x, self.y, self.z, self.vx, self.vy, self.vz]

    ##### Функция для вичисления частных производных по кеплеровым элементам #####
    def derivatives(self, t=t0, dt=None):
        a, e, i, W, w, M0 = self.get('kep')
        mu = self.mu
        t0 = self.t0

        A = [[cos(W), -sin(W), 0],
             [sin(W), cos(W), 0],
             [0, 0, 1]]
        A = np.matrix(A)

        B = [[1, 0, 0],
             [0, cos(i), -sin(i)],
             [0, sin(i), cos(i)]]
        B = np.matrix(B)

        C = [[cos(w), -sin(w), 0],
             [sin(w), cos(w), 0],
             [0, 0, 1]]
        C = np.matrix(C)

        R = A * B * C

        n = sqrt(mu * a ** (-3))

        if (dt):
            M = M0 + n * dt
        else:
            M = M0 + n * (t - t0)

        M = reduce(M)

        # Численное решение уравнения Кеплера
        E = M
        while (abs(E - e * sin(E) - M) > 1e-12):
            E = E - (E - e * sin(E) - M) / (1 - e * cos(E))

        q = np.matrix([a * (cos(E) - e), a * sqrt(1 - e ** 2) * sin(E), 0])

        reduced = n * (t - t0)  # M - M0

        # частные производные
        dksi_da = cos(E) - e + 3 / 2 * (reduced * sin(E)) / (1 - e * cos(E))
        deta_da = sqrt(1 - e ** 2) * (sin(E) - 3 / 2 * (reduced * cos(E)) / (1 - e * cos(E)))
        da = np.matrix([dksi_da, deta_da, 0])
        dxyz_da = R * da.T

        dksi_de = -a * (1 + (sin(E) ** 2) / (1 - e * cos(E)))
        deta_de = a * sqrt(1 - e ** 2) * sin(E) * ((cos(E) / (1 - e * cos(E))) - (e / (1 - e ** 2)))
        de = np.matrix([dksi_de, deta_de, 0])
        dxyz_de = R * de.T

        dksi_dM0 = - (a * sin(E)) / (1 - e * cos(E))
        deta_dM0 = (a * (sqrt(1 - e ** 2)) * cos(E)) / (1 - e * cos(E))
        dM0 = np.matrix([dksi_dM0, deta_dM0, 0])
        dxyz_M0 = R * dM0.T

        dB = [[0, 0, 0],
              [0, -sin(i), -cos(i)],
              [0, cos(i), -sin(i)]]
        dB = np.matrix(dB)

        dA = [[-sin(W), -cos(W), 0],
              [cos(W), -sin(W), 0],
              [0, 0, 0]]
        dA = np.matrix(dA)

        dC = [[-sin(w), -cos(w), 0],
              [cos(w), -sin(w), 0],
              [0, 0, 0]]
        dC = np.matrix(dC)

        dxyz_di = A * dB * C * q.T
        dxyz_dw = A * B * dC * q.T
        dxyz_dW = dA * B * C * q.T
        dxyz_da_dM0 = np.concatenate([dxyz_da, dxyz_de, dxyz_di, dxyz_dw, dxyz_dW, dxyz_M0], axis=1)
        return dxyz_da_dM0

    ##### Функция получения кеплеровых элементов #####
    def findOrbit(self):
        q = [self.x, self.y, self.z]
        v = [self.vx, self.vy, self.vz]

        r = sqrt(dot(q, q))
        V = sqrt(dot(v, v))
        s = dot(q, v)
        h = cross(q, v)

        k = sqrt(self.mu)

        # Большая полуось
        a = 1 / abs(2. / r - V ** 2 / k ** 2)

        # Эксцентрисистет
        e = sqrt(s * s / (k * k * a) + (1 - r / a) ** 2)

        # Средняя аномалия:
        dy = s / (e * k * sqrt(a))
        dx = (a - r) / (a * e)
        E0 = arctan2(dy, dx)
        M0 = E0 - e * sin(E0)

        # Долгота восходящего узла:
        W = reduce(arctan2(h[0], -h[1]))

        # Наклонение
        i = arctan2(sqrt(h[0] ** 2 + h[1] ** 2), h[2])

        # Аргумент перицентра:
        p = a * (1 - e ** 2)

        dy = sqrt(p) * s
        dx = k * (p - r)
        vv = arctan2(dy, dx)

        if (sin(i) != 0):
            dy = self.z / sin(i)
            dx = self.x * cos(W) + self.y * sin(W)
            uu = arctan2(dy, dx)
        else:
            uu = 0

        w = uu - vv
        while (w < 0):
            w += 2 * pi

        self.a, self.e, self.i, self.W, self.w, self.M0 = a, e, i, W, w, M0
        return [self.a, self.e, self.i, self.W, self.w, self.M0]

    ##### Функция определения частных производных по alpha, delta #####
    def derivatives2(self, t=t0, dt=None):
        xr = aster.x - earth.x
        yr = aster.y - earth.y
        zr = aster.z - earth.z

        r = sqrt(xr ** 2 + yr ** 2 + zr ** 2)

        alpha = reduce(np.arctan2(yr, xr))
        delta = arcsin(zr / r)

        dalpha_dx = -sin(alpha) / (r * cos(delta))
        ddelta_dx = -cos(alpha) * sin(delta) / r

        dalpha_dy = cos(alpha) / (r * cos(delta))
        ddelta_dy = -sin(alpha) * sin(delta) / r

        dalpha_dz = 0
        ddelta_dz = cos(delta) / r

        dalpha_dq = np.matrix([dalpha_dx, dalpha_dy, dalpha_dz])
        ddelta_dq = np.matrix([ddelta_dx, ddelta_dy, ddelta_dz])
        dalpha_dq_ddelta_dq = np.concatenate([dalpha_dq, ddelta_dq])
        return dalpha_dq_ddelta_dq

    ##### Функция определения частных производных по alpha, delta со скоростями #####
    def derivatives3(self, t=t0, dt=None):
        a, e, i, W, w, M0 = self.get('kep')
        mu = self.mu
        t0 = self.t0

        A = [[cos(W), -sin(W), 0],
             [sin(W), cos(W), 0],
             [0, 0, 1]]
        A = np.matrix(A)

        B = [[1, 0, 0],
             [0, cos(i), -sin(i)],
             [0, sin(i), cos(i)]]
        B = np.matrix(B)

        C = [[cos(w), -sin(w), 0],
             [sin(w), cos(w), 0],
             [0, 0, 1]]
        C = np.matrix(C)

        R = A * B * C

        n = sqrt(mu * a ** (-3))

        if (dt):
            M = M0 + n * dt
        else:
            M = M0 + n * (t - t0)
        M = reduce(M)

        # Численное решение уравнения Кеплера
        E = M
        while (abs(E - e * sin(E) - M) > 1e-12):
            E = E - (E - e * sin(E) - M) / (1 - e * cos(E))

        q = np.matrix([a * (cos(E) - e), a * sqrt(1 - e ** 2) * sin(E), 0])
        dq = np.matrix([-a * n * sin(E) / (1 - e * cos(E)), a * sqrt(1 - e ** 2) * cos(E) * n / (1 - e * cos(E)), 0])
        reduced = n * (t - t0)  # M - M0

        # частные производные
        dksi_da = cos(E) - e + 3 / 2 * (reduced * sin(E)) / (1 - e * cos(E))
        ddksi_da = 1 / 2 * n * (sin(E) / (1 - e * cos(E))) + 3 / 2 * n * reduced * (
                (cos(E) - e) / (1 - e * cos(E)) ** 3)
        deta_da = sqrt(1 - e ** 2) * (sin(E) - 3 / 2 * (reduced * cos(E)) / (1 - e * cos(E)))
        ddeta_da = - (n * sqrt(1 - e * e) * cos(E)) / (2 * (1 - e * cos(E))) + 3 / 2 * n * sqrt(
            1 - e ** 2) * reduced * (sin(E) / (1 - e * cos(E)) ** 3)
        da = np.matrix([dksi_da, deta_da, 0])
        dda = np.matrix([ddksi_da, ddeta_da, 0])
        dxyz_da = R * da.T
        dxyz_dda = R * dda.T

        dksi_de = -a * (1 + (sin(E) ** 2) / (1 - e * cos(E)))
        ddksi_de = - ((a * n * sin(E)) / (1 - e * cos(E)) ** 2) * (cos(E) + (cos(E) - e) / (1 - e * cos(E)))
        deta_de = a * sqrt(1 - e ** 2) * sin(E) * ((cos(E) / (1 - e * cos(E))) - (e / (1 - e ** 2)))
        ddeta_de = - (a * e * n * cos(E)) / (sqrt(1 - e ** 2) * (1 - e * cos(E))) - (
                (a * n * sqrt(1 - e ** 2)) / (1 - e * cos(E)) ** 2) * (
                           sin(E) * sin(E) - cos(E) * (cos(E) - e) / (1 - e * cos(E)))
        de = np.matrix([dksi_de, deta_de, 0])
        dde = np.matrix([ddksi_de, ddeta_de, 0])
        dxyz_de = R * de.T
        dxyz_dde = R * dde.T

        dksi_dM0 = - (a * sin(E)) / (1 - e * cos(E))
        ddksi_dM0 = - (n * a * (cos(E) - e)) / (1 - e * cos(E)) ** 3
        deta_dM0 = (a * (sqrt(1 - e ** 2)) * cos(E)) / (1 - e * cos(E))
        ddeta_dM0 = - (n * a * sqrt(1 - e ** 2) * sin(E)) / (1 - e * cos(E)) ** 3
        dM0 = np.matrix([dksi_dM0, deta_dM0, 0])
        dxyz_M0 = R * dM0.T
        ddM0 = np.matrix([ddksi_dM0, ddeta_dM0, 0])
        ddxyz_M0 = R * ddM0.T

        dB = [[0, 0, 0],
              [0, -sin(i), -cos(i)],
              [0, cos(i), -sin(i)]]
        dB = np.matrix(dB)

        dA = [[-sin(W), -cos(W), 0],
              [cos(W), -sin(W), 0],
              [0, 0, 0]]
        dA = np.matrix(dA)

        dC = [[-sin(w), -cos(w), 0],
              [cos(w), -sin(w), 0],
              [0, 0, 0]]
        dC = np.matrix(dC)

        dxyz_di = A * dB * C * q.T
        dxyz_dw = A * B * dC * q.T
        dxyz_dW = dA * B * C * q.T
        dxyz_ddi = A * dB * C * dq.T
        dxyz_ddw = A * B * dC * dq.T
        dxyz_ddW = dA * B * C * dq.T

        dxyz_da_dM0 = np.concatenate([dxyz_da, dxyz_de, dxyz_di, dxyz_dw, dxyz_dW, dxyz_M0], axis=1)
        dxyz_dda_ddM0 = np.concatenate([dxyz_dda, dxyz_dde, dxyz_ddi, dxyz_ddw, dxyz_ddW, ddxyz_M0], axis=1)
        dxyz_da_dM0_dda_ddM0 = np.concatenate([dxyz_da_dM0, dxyz_dda_ddM0])
        return dxyz_da_dM0_dda_ddM0

    ##### Процедуры установки новых параметров #####
    def set(self, settings, type='kep'):
        if (type == 'kep'):
            self.a, self.e, self.i, self.W, self.w, self.M0 = settings
            self.cartesian()

        if (type == 'qv'):
            self.x, self.y, self.z, self.vx, self.vy, self.vz = settings
            self.findOrbit()

    ##### Процедуры возвращающие параметры: ######
    def get(self, type='kep'):
        if (type == 'kep'):
            return [self.a, self.e, self.i, self.W, self.w, self.M0]


##### Создание объектов класса #####

aster = Sat([a, e, i, W, w, M0], ptype='kep', mu=mu_solar_au, t0=t0)
earth = Sat([a2, e2, i2, W2, w2, M02], ptype='kep', mu=mu_solar_au, t0=t0)

############## Метод наименьших квадратов ###############

##### Формирование моментов времени ######

tm = [2453974.77504250, 2453974.79553250, 2453974.80586250, 2453975.80620250, 2453975.82639251, 2453975.84370251,
      2453995.76306264, 2453995.77673264, 2453995.79040264, 2453995.80409264, 2453995.82592264, 2453995.82958264]

##### Формируем ошибки наблюдений #####
ErrAlpha = []
ErrAlpha = (np.random.normal(0, 1, n2)) * rad / 3600

ErrDelta = []
ErrDelta = (np.random.normal(0, 1, n2)) * rad / 3600

##### Формируем альфа, дельта #####

alpha = []
delta = []
for j in range(n2):
    AstRaz = aster.cartesian(tm[j])
    ZemRaz = earth.cartesian(tm[j])
    RazX = aster.x - earth.x
    RazY = aster.y - earth.y
    RazZ = aster.z - earth.z

    RRaz = sqrt(RazX ** 2 + RazY ** 2 + RazZ ** 2)

    alph = reduce(np.arctan2(RazY, RazX))
    alpha.append(alph)
    delt = arcsin(RazZ / RRaz)
    delta.append(delt)

AlphaObs = ErrAlpha + alpha
DeltaObs = ErrDelta + delta

XZAI = aster.derivatives(t0)
XZDA = aster.derivatives2(t0)

##### Основной цикл #####
Decart = np.matrix(aster.cartesian(t0)).T

while True:

    matrixL = np.zeros(shape=(2 * n2, 1))
    matrixR = np.zeros(shape=(2 * n2, 6))
    RTL = np.zeros(6)
    RTR = np.zeros(shape=(6, 6))
    fq = 0

    for j in range(n2):
        XZAI = aster.derivatives(tm[j])
        AstRaz = aster.cartesian(tm[j])
        ZemRaz = earth.cartesian(tm[j])

        RazX = aster.x - earth.x
        RazY = aster.y - earth.y
        RazZ = aster.z - earth.z
        Decart = np.matrix(aster.cartesian(t0)).T

        RRaz = sqrt(RazX ** 2 + RazY ** 2 + RazZ ** 2)

        alph = reduce(np.arctan2(RazY, RazX))
        delt = arcsin(RazZ / RRaz)

        dalph_dx = -sin(alph) / (RRaz * cos(delt))
        ddelt_dx = -cos(alph) * sin(delt) / RRaz

        dalph_dy = cos(alph) / (RRaz * cos(delt))
        ddelt_dy = -sin(alph) * sin(delt) / RRaz

        dalph_dz = 0
        ddelt_dz = cos(delt) / RRaz

        # Нормальная матрица
        for k in range(6):
            for p in range(6):
                RTR[k][p] = RTR[k][p] + (dalph_dx * XZAI[0, k] + dalph_dy * XZAI[1, k]) * (
                        dalph_dx * XZAI[0, p] + dalph_dy * XZAI[1, p]) * (cos(DeltaObs[j])) * (cos(DeltaObs[j])) + (
                                    ddelt_dx * XZAI[0, k] + ddelt_dy * XZAI[1, k] + ddelt_dz * XZAI[2, k]) * (
                                    ddelt_dx * XZAI[0, p] + ddelt_dy * XZAI[1, p] + ddelt_dz * XZAI[2, p])

        for i in range(6):
            RTL[i] = RTL[i] + (dalph_dx * XZAI[0, i] + dalph_dy * XZAI[1, i]) * (alph - AlphaObs[j]) * (
                cos(DeltaObs[j])) * (cos(DeltaObs[j])) + (
                             ddelt_dx * XZAI[0, i] + ddelt_dy * XZAI[1, i] + ddelt_dz * XZAI[2, i]) * (
                             delt - DeltaObs[j])

        matrixL[2 * j - 1][0] = (alph - AlphaObs[j]) * cos(DeltaObs[j])
        matrixL[2 * j][0] = delt - DeltaObs[j]

        for i in range(6):
            matrixR[2 * j - 1][i] = (dalph_dx * XZAI[0, i] + dalph_dy * XZAI[1, i]) * cos(DeltaObs[j])
            matrixR[2 * j][i] = (ddelt_dx * XZAI[0, i] + ddelt_dy * XZAI[1, i] + ddelt_dz * XZAI[2, i])

        fq = fq + ((alph - AlphaObs[j]) * cos(DeltaObs[j])) ** 2 + (delt - DeltaObs[j]) ** 2

    ObrRTR = np.linalg.inv(RTR)

    RORT = (aster.derivatives3(t0)) * ObrRTR * (aster.derivatives3(t0)).T  # P * ObrRTR * P.T
    RObrL = (matrixR * np.linalg.inv(aster.derivatives3(t0))).T * matrixL
    Mod = RORT * RObrL
    MDecart = Decart - Mod

    sigma = sqrt(fq / (2 * n2 - 6))
    matrixD = sigma ** 2 * RORT

    Decarton = Sat(MDecart.squeeze().tolist()[0], ptype='qv', mu=mu_solar_au, t0=t0)
    orbit = Decarton.get('kep')
    aster = Sat(orbit, ptype='kep', mu=mu_solar_au, t0=t0)
    if np.max(np.abs(MDecart - Decart)) < 1e-10:
        break
ssgm = sigma

##### ММК(заполнение точек по всему объему эллипсода) #####

XE = []
YE = []
ZE = []
for i in range(kt):
    AHol = np.linalg.cholesky(matrixD)
    randNumb = np.matrix([random.normal(0, 1) for i in range(6)]).T
    DecartELP = MDecart + AHol * randNumb
    XE.append(DecartELP[0, 0])
    YE.append(DecartELP[1, 0])
##### формирование эллипсоида по граничной поверхности #####
AH = []
XX = []
YY = []
ZZ = []
AH2 = []

for i in range(kt):
    randNumb22 = np.matrix([random.normal(0, sigma ** 2) for i in range(6)]).T
    AHol2 = np.linalg.cholesky(matrixD)
    AHR2 = AHol * randNumb22
    for j in range(6):
        AHR2[j, 0] = (kk / sqrt(
            (randNumb22[0, 0] ** 2) + (randNumb22[1, 0] ** 2) + (randNumb22[2, 0] ** 2) + (randNumb22[3, 0] ** 2) + (
                    randNumb22[4, 0] ** 2) + (randNumb22[5, 0] ** 2))) * AHR2[j, 0]
    MDAHR2 = MDecart + AHR2
    AH2.append(MDAHR2)

for i in range(kt):
    randNumb2 = np.matrix([random.normal(0, 1) for i in range(6)]).T
    AHol = np.linalg.cholesky(matrixD)
    AHR = AHol * randNumb2
    for j in range(6):
        AHR[j, 0] = (kk / sqrt(
            (randNumb2[0, 0] ** 2) + (randNumb2[1, 0] ** 2) + (randNumb2[2, 0] ** 2) + (randNumb2[3, 0] ** 2) + (
                    randNumb2[4, 0] ** 2) + (randNumb2[5, 0] ** 2))) * AHR[j, 0]
    MDAHR = MDecart + AHR
    XX.append(MDAHR[0, 0])
    YY.append(MDAHR[1, 0])
    ZZ.append(MDAHR[2, 0])
    AH.append(MDAHR)

##### Метод Милани #####
h = 0.1
S = np.random.normal(0, 1)
Sn = 0
MX = []
MY = []
MZ = []
for i in tqdm.tqdm(range(kt)):

    while True:
        Decarton = Sat(MDecart.squeeze().tolist()[0], ptype='qv', mu=mu_solar_au, t0=t0)
        orbit = Decarton.get('kep')
        aster = Sat(orbit, ptype='kep', mu=mu_solar_au, t0=t0)

        matrixR = np.zeros(shape=(2 * n2, 6))
        RTR = np.zeros(shape=(6, 6))

        for j in range(n2):
            XZAI = aster.derivatives(tm[j])
            AstRaz = aster.cartesian(tm[j])
            ZemRaz = earth.cartesian(tm[j])

            RazX = aster.x - earth.x
            RazY = aster.y - earth.y
            RazZ = aster.z - earth.z
            Decart = np.matrix(aster.cartesian(t0)).T

            RRaz = sqrt(RazX ** 2 + RazY ** 2 + RazZ ** 2)

            alph = reduce(np.arctan2(RazY, RazX))
            delt = arcsin(RazZ / RRaz)

            dalph_dx = -sin(alph) / (RRaz * cos(delt))
            ddelt_dx = -cos(alph) * sin(delt) / RRaz

            dalph_dy = cos(alph) / (RRaz * cos(delt))
            ddelt_dy = -sin(alph) * sin(delt) / RRaz

            dalph_dz = 0
            ddelt_dz = cos(delt) / RRaz

            # Нормальная матрица
            for k in range(6):
                for p in range(6):
                    RTR[k][p] = RTR[k][p] + (dalph_dx * XZAI[0, k] + dalph_dy * XZAI[1, k]) * (
                            dalph_dx * XZAI[0, p] + dalph_dy * XZAI[1, p]) * (cos(DeltaObs[j])) * (cos(DeltaObs[j])) + (
                                        ddelt_dx * XZAI[0, k] + ddelt_dy * XZAI[1, k] + ddelt_dz * XZAI[2, k]) * (
                                        ddelt_dx * XZAI[0, p] + ddelt_dy * XZAI[1, p] + ddelt_dz * XZAI[2, p])

        ObrRTR = np.linalg.inv(RTR)

        RORT = (aster.derivatives3(t0)) * ObrRTR * (aster.derivatives3(t0)).T  # P * ObrRTR * P.T

        matrixD = ssgm ** 2 * RORT

        DSobZ, DSobV = np.linalg.eig(matrixD)

        if S > Sn:
            NMDecart = MDecart + h * max(DSobZ) * ((DSobV.T[0]).T)
            Sn = Sn + h
            MDecart = NMDecart
        else:
            NMDecart = MDecart + S * max(DSobZ) * ((DSobV.T[0]).T)
            break
    MDecart = NMDecart
    MX.append(NMDecart[0, 0])
    MY.append(NMDecart[1, 0])

plt.scatter(MX, MY, marker='x', color='blue')
plt.scatter(XX, YY, marker='o', color='yellow')

plt.xlim((min(MX) - 10 ** -7), (max(MX) + 10 ** -7))
plt.ylim((min(MY) - 10 ** -7), (max(MY) + 10 ** -7))
plt.show()
