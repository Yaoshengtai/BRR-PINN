import torch

# Generate different types of collocation points
# Use 'next(generator)' to iteration

def generator_1dspatial(size, x_min, x_max, device,random=False,bound=True):
# Return a generator that generates 1D points range from x_min to x_max

    # param size: Number of points to generated
    # type size: int

    # param x_min: Lower bound of x
    # type x_min: float

    # param x_max: Upper bound of x
    # type x_max: float

    # param device: where to generate points (e.g. cpu/gpu/...)
    # type device: torch.device

    # param random:
    #     - If set to False, return equally spaced points range from x_min to x_max
    #     - If set to True, return points generated randomly
    # type random: bool

    # param bound:
    #     - If set to False, generate points without the bound of the range (point x_min and x_max)
    #     - If set to True, generate points with the bound of the range (point x_min and x_max)
    # type bound: bool

    seg_len = (x_max-x_min) / size
    if bound==True:
        linspace_lo = x_min
        linspace_hi = x_max
    else:
        linspace_lo = x_min + seg_len*0.5
        linspace_hi = x_max - seg_len*0.5
    center = torch.linspace(linspace_lo, linspace_hi, size).to(device)
    noise_lo = -seg_len*0.5
    while True:
        center = torch.linspace(linspace_lo, linspace_hi, size).to(device)
        if random:
            noise = seg_len*torch.rand(size) + noise_lo
            noise=noise.to(device)
            noise[0]=0
            noise[-1]=0
            yield center + noise
        else:
            yield center


def generator_2dspatial_segment(size, start, end, device,random=False, bound=True):
# Return a generator that generates 1D points range from x_min to x_max

    # param size: Number of points to generated
    # type size: int

    # param start: start point of the 2D line
    # type start: tuple: (coordinate x, coordinate y): (float, float)

    # param end: end point of the 2D line
    # type end: tuple: (coordinate x, coordinate y): (float, float)

    # param device: where to generate points (e.g. cpu/gpu/...)
    # type device: torch.device

    # param random:
    #     - If set to False, return equally spaced points
    #     - If set to True, return points generated randomly
    # type random: bool

    # param bound:
    #     - If set to False, generate points without the bound of the range (point start and end)
    #     - If set to True, generate points with the bound of the range (point start and end)
    # type bound: bool

    x1, y1 = start
    x2, y2 = end
    step = 1./size
    noise_lo = -step*0.5
    while True:
        if bound:
            center = torch.linspace(0. , 1. , size).to(device)
        else:
            center = torch.linspace(0. + 0.5*step, 1. - 0.5*step, size).to(device)
        if random:
            noise = step*torch.rand(size) + noise_lo
            noise=noise.to(device)
            center = center.to(device) + noise
        yield x1 + (x2-x1)*center, y1 + (y2-y1)*center


def generator_2dspatial_rectangle(size, x_min, x_max, y_min, y_max, device,random=False,bound=True):
# Return a generator that generates points in a 2D rectangle

    # param size: Number of points to generated
    # type size: int

    # param x_min, x_max: the x range of the rectangle
    # type x_min, x_max: float

    # param y_min, y_max: the y range of the rectangle
    # type y_min, y_max: float

    # param device: where to generate points (e.g. cpu/gpu/...)
    # type device: torch.device

    # param random:
    #     - If set to False, return equally spaced points
    #     - If set to True, return points generated randomly
    # type random: bool

    # param bound:
    #     - If set to False, generate points without the bound of the range (point start and end)
    #     - If set to True, generate points with the bound of the range (point start and end)
    # type bound: bool

    x_size, y_size = size
    x_generator = generator_1dspatial(x_size, x_min, x_max, device,random,bound)
    y_generator = generator_1dspatial(y_size, y_min, y_max, device,random,bound)
    while True:
        x = next(x_generator).to(device)
        y = next(y_generator).to(device)
        xy = torch.cartesian_prod(x, y)
        xx = torch.squeeze(xy[:, 0])
        yy = torch.squeeze(xy[:, 1])
        yield xx, yy