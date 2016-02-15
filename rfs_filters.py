#def makeRFSfilters():
# Returns the RFS filter bank of size 49x49x38 in F. The MR8, MR4 and
# MRS4 sets are all derived from this filter bank. To convolve an
# image I with the filter bank you can either use the matlab function
# conv2, i.e. responses(:,:,i)=conv2(I,F(:,:,i),'valid'), or use the
# Fourier transform.
#%%
import numpy

def makefilter(scale, phasex, phasey, pts, sup):
    gx = gauss1d(3 * scale, 0, pts[0, :], phasex)
    gy = gauss1d(scale, 0, pts[1, :], phasey)
    f = normalise((gx * gy).reshape((sup, sup)))
    
    return f

def gauss1d(sigma, mean, x, ord):
# Function to compute gaussian derivatives of order 0 <= ord < 3
# evaluated at x.
    x = x - mean;
    num = x * x;
    variance = sigma**2;
    denom = 2 * variance; 
    g = numpy.exp(-num / denom) / numpy.sqrt(numpy.pi * denom)
    
    if ord == 0:
        return g
    elif ord == 1:
        return -g * (x / variance)
    else:
        return g * ((num - variance) / variance**2);

def normalise(f):
    f = f - numpy.mean(f);
    f = f / sum(abs(f));
    return f

def make(sup = 49, scalex = [1, 2, 4], norient = 6):
    # sup == Support of the largest filter (must be odd)
    # scalex == Sigma_{x} for the oriented filters
    # norient == Number of orientations

    nrotinv = 2
    
    nbar = len(scalex) * norient
    nedge = len(scalex) * norient
    nf = nbar + nedge + nrotinv
    F = numpy.zeros((sup, sup, nf))
    hsup = (sup - 1) / 2

    x, y = numpy.meshgrid(numpy.arange(-hsup, hsup + 1), numpy.arange(hsup, -hsup - 1, -1))
    
    orgpts = numpy.array([x.flatten(), y.flatten()])

    count = 0;
    for scale in scalex:
        for orient in range(norient):
            angle = numpy.pi * orient / norient  # Not 2pi as filters have symmetry
            c = numpy.cos(angle)
            s = numpy.sin(angle)
            
            rotpts = numpy.array([[c, -s], [s, c]]).dot(orgpts)
            
            F[:, :, count] = makefilter(scale, 0, 1, rotpts, sup);
            F[:, :, count + nedge] = makefilter(scale, 0, 2, rotpts, sup);
            
            count=count+1;

    r = orgpts[0, :]**2 + orgpts[1, :]**2
    sigma = 10.0
    
    F[:, :, nbar + nedge] = normalise(((1.0 / numpy.sqrt(2 * numpy.pi * sigma**2)) * numpy.exp(-r / (2 * sigma**2))).reshape(sup, sup))
    F[:, :, nbar + nedge + 1] = normalise(((r - 2 * sigma**2) / (sigma ** 4) * numpy.exp(-r / (2 * sigma**2))).reshape(sup, sup))

    return F
