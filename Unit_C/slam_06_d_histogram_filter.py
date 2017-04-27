# Histogram implementation of a bayes filter - combines
# convolution and multiplication of distributions, for the
# movement and measurement steps.
# 06_d_histogram_filter
# Claus Brenner, 28 NOV 2012
from pylab import plot, show, ylim
from distribution import *

def move(distribution, delta):
    """Returns a Distribution that has been moved (x-axis) by the amount of
       delta."""
    return Distribution(distribution.offset + delta, distribution.values)


def convolve(a, b):
    """Convolve distribution a and b and return the resulting new distribution."""

    # --->>> Put your code here.
    dist_lst = []
    offs = (a.offset + b.offset)
    for a_val in a.values:
        res = []
        for b_val in b.values:
            res.append(a_val * b_val)
        dist_lst.append(Distribution(offs, res))
        offs += 1
    c = Distribution.sum(dist_lst)

    return c  # Replace this by your own result.


def multiply(a, b):
    """Multiply two distributions and return the resulting distribution."""

    start = min([a.start(), b.start()])
    stop = max([a.stop(), b.stop()])
    mul_dist = []
    for ix in range(start, stop):
        mul_dist.append(a.value(ix) * b.value(ix))

    c = Distribution(start, mul_dist)
    Distribution.normalize(c)

    return c  # Modify this to return your result.

# --->>> Copy your convolve(a, b) and multiply(a, b) functions here.



if __name__ == '__main__':
    arena = (0,220)

    # Start position. Exactly known - a unit pulse.
    start_position = 10
    position = Distribution.unit_pulse(start_position)
    plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
         linestyle='steps')

    # Movement data.
    controls  =    [ 20 ] * 10

    # Measurement data. Assume (for now) that the measurement data
    # is correct. - This code just builds a cumulative list of the controls,
    # plus the start position.
    p = start_position
    measurements = []
    for c in controls:
        p += c
        measurements.append(p)

    # This is the filter loop.
    for i in xrange(len(controls)):
        # Move, by convolution. Also termed "prediction".
        control = Distribution.triangle(controls[i], 10)
        position = convolve(position, control)
        plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
             color='b', linestyle='steps')

        # Measure, by multiplication. Also termed "correction".
        measurement = Distribution.triangle(measurements[i], 10)
        position = multiply(position, measurement)
        plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
             color='r', linestyle='steps')

    show()
