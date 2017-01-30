import pylab
import numpy

x = numpy.linspace(-8,8,100) # 100 linearly spaced numbers
y = 1/(1+numpy.exp(-x)) # computing the values of sin(x)/x

# compose plot
pylab.plot(x,y) # sin(x)/x
ax = pylab.axes()
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
pylab.show() # show the plot