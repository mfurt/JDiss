from consts import mu_solar_au
from main import Sat
from utils import ephem

a2 = 0.99389833340476404507247
e2 = 0.016071407248214105899401
i2 = 0.40907718387495823273769
w2 = 1.8012124448649747146848
W2 = 6.2831798469319858809286
M02 = 5.97634233828
t0 = 2453356.2387180547230

earth = Sat([a2, e2, i2, W2, w2, M02], ptype='kep', mu=mu_solar_au, t0=t0)
ephemeris_earth = ephem(t0, 'earth')

print('Чёрный: ')
print(earth.cartesian())
print('Эфемериды')
print(list(ephemeris_earth))
earth.set(list(ephemeris_earth), type='qv')


print('a\t%.8f\t%.8f\t%.2e' % (earth.a, a2, a2-earth.a))
print('e\t%.8f\t%.8f\t%.2e' % (earth.e, e2, e2-earth.e))
print('i\t%.8f\t%.8f\t%.2e' % (earth.i, i2, i2-earth.i))
print('w\t%.8f\t%.8f\t%.2e' % (earth.w, w2, w2-earth.w))
print('W\t%.8f\t%.8f\t%.2e' % (earth.W, W2, a2-earth.W))
print('M0\t%.8f\t%.8f\t%.2e' % (earth.M0, M02, M02-earth.M0))