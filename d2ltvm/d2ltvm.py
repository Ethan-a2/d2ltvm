import sys
import tvm
from tvm import te
import tvm.topi as topi
from tvm.topi import nn  # For common neural network operators
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt
from IPython import display as ipython_display # Alias to avoid conflict if 'display' is used elsewhere.

# This line ensures that within other functions in this module,
# calls like get_abc() resolve correctly to get_abc().
# This mimics the original d2lbook generated structure.
d2ltvm = sys.modules[__name__] if '__name__' in sys.modules else type('d2ltvm_module', (object,), {})()

try:
    import mxnet as mx
except ImportError:
    pass # MXNet is optional

# Defined in file: ./chapter_getting_started/vector_add.md
def get_abc(shape: tuple, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c


# Defined in file: ./chapter_getting_started/vector_add.md
def vector_add(n):
    """TVM expression for vector add"""
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

# Defined in file: ./chapter_getting_started/from_mxnet.md
def image_preprocessing(image: np.ndarray) -> np.ndarray:
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype('float32')


# Defined in file: ./chapter_common_operators/broadcast_add.md
def broadcast_add(shape1: tuple, shape2: tuple):
    """Broadcast add between two 2-dimensional tensors using topi.

    shape1, shape2 : the shapes of the input tensors
    """
    assert len(shape1) == 2 and len(shape2) == 2, \
        "broadcast tensors should both be 2-dimension"
    for i in range(len(shape1)):
        assert shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1, \
            "tensor shapes do not fit for broadcasting"

    A = te.placeholder(shape1, name='A')
    B = te.placeholder(shape2, name='B')
    C = topi.add(A, B) # topi.add handles broadcasting automatically
    return A, B, C


# Defined in file: ./chapter_common_operators/broadcast_add.md
def get_bcast_data(shape1: tuple, shape2: tuple, constructor=None):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b

    shape1, shape2: shapes of input tensors
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")

    # Determine output shape based on broadcasting rules
    out_shape = []
    for d1, d2 in zip(shape1, shape2):
        if d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        else:
            out_shape.append(d1) # Must be equal if neither is 1
    c = np.empty(tuple(out_shape), dtype='float32')

    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c


# Defined in file: ./chapter_common_operators/matmul.md
def matmul(n: int, m: int, l: int):
    """Return the computing expression of matrix multiplication using topi.
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = topi.matmul(A, B)
    return A, B, C


# Defined in file: ./chapter_common_operators/conv.md
def padding(X: tvm.te.Tensor, ph: int, pw: int, val=0) -> tvm.te.Tensor:
    """Pad X with the given value in 2-D or N-D

    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2, "Input tensor must have at least 2 dimensions"
    nh, nw = X.shape[-2], X.shape[-1]

    # Handle arbitrary leading dimensions
    pad_shape = (*X.shape[0:-2], nh + ph * 2, nw + pw * 2)

    PaddedX = te.compute(
        pad_shape,
        lambda *i: te.if_then_else(
            te.any(i[-2] < ph, i[-2] >= nh + ph, i[-1] < pw, i[-1] >= nw + pw),
            val, X[i[:-2] + (i[-2] - ph, i[-1] - pw)])
        , name='PaddedX')
    return PaddedX


# Defined in file: ./chapter_common_operators/conv.md
def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1


# Defined in file: ./chapter_common_operators/conv.md
def conv(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and width
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (oc, oh, ow),
        lambda c, i, j: te.sum(
            PaddedX[ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
            axis=[ric, rkh, rkw]), name='Y')
    return X, K, Y, PaddedX


# Defined in file: ./chapter_common_operators/conv.md
def conv_mxnet(data: 'mx.nd.NDArray', weight: 'mx.nd.NDArray', bias: 'mx.nd.NDArray',
               out: 'mx.nd.NDArray', k: int, p: int, s: int):
    """Convolution in MXNet"""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    mx.nd.Convolution(data, weight, bias, kernel=(k,k), stride=(s,s),
                      pad=(p,p), num_filter=out.shape[1], out=out)


# Defined in file: ./chapter_common_operators/depthwise_conv.md
def get_conv_data(oc, ic, n, k, p=0, s=1, constructor=None, conv_type='direct'):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output 
    tensor with the shapes specified by input arguments.

    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    conv_type: either direct 2D or depthwise, default direct
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(ic, n, n)).astype('float32')
    ic_weight = ic
    if conv_type == 'depthwise':
        ic_weight = 1
    weight = np.random.normal(size=(oc, ic_weight, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out


# Defined in file: ./chapter_common_operators/depthwise_conv.md
def depthwise_conv(ic: int, nh: int, nw: int, kh: int, kw: int, ph: int=0, pw: int=0, sh: int=1, sw: int=1):
    """Depthwise Convolution using topi.nn.depthwise_conv2d_nchw

    ic : number of channels for both input and output
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    X_pl = te.placeholder((ic, nh, nw), name='X', dtype='float32')
    K_pl = te.placeholder((ic, 1, kh, kw), name='K', dtype='float32')

    data = topi.expand_dims(X_pl, axis=0, num_dims=1) # (1, ic, nh, nw)
    kernel = K_pl # Already 4D (ic, 1, kh, kw)

    conv_out = nn.depthwise_conv2d_nchw(
        data, kernel,
        stride=(sh, sw),
        padding=(ph, pw),
        dilation=(1, 1),
        out_dtype='float32'
    )
    Y_pl = topi.squeeze(conv_out, axis=[0]) # (ic, oh, ow)

    # Return symbolic PaddedX to match original function signature
    symbolic_padded_input = padding(X_pl, ph, pw) # Use  prefix as in original
    return X_pl, K_pl, Y_pl, symbolic_padded_input


# Defined in file: ./chapter_common_operators/depthwise_conv.md
def get_conv_data_mxnet(oc: int, ic: int, n: int, k: int, p: int, s: int, ctx: str='cpu', conv_type: str='direct'):
    """Return MXNet data for convolution benchmark using get_conv_data."""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    ctx_obj = getattr(mx, ctx)()
    data, weight, out = get_conv_data(oc, ic, n, k, p, s, # Use  prefix as in original
                                      constructor=lambda x: mx.nd.array(x, ctx=ctx_obj),
                                      conv_type=conv_type)
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    bias = mx.nd.zeros(out.shape[1], ctx=ctx_obj)
    return data, weight, bias, out


# Defined in file: ./chapter_common_operators/depthwise_conv.md
def depthwise_conv_mxnet(data: 'mx.nd.NDArray', weight: 'mx.nd.NDArray', bias: 'mx.nd.NDArray',
                         out: 'mx.nd.NDArray', k: int, p: int, s: int):
    """Depthwise Convolution in MXNet"""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    mx.nd.Convolution(data, weight, bias, kernel=(k,k), stride=(s,s),
                      pad=(p,p), num_filter=out.shape[1],
                      out=out, num_group=weight.shape[0])


# Defined in file: ./chapter_common_operators/pooling.md
def pool(pool_type, c, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2D pooling
    
    pool_type: pooling type, 'max' or 'avg'
    c : channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((c, nh, nw), name='X')
    
    
    if pool_type == 'max':
        PaddedX = padding(X, ph, pw, val=te.min_value(X.dtype)) \
            if ph * pw != 0 else X
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.max(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_max", name='PoolMax')
    elif pool_type == 'avg':
        PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
        tsum = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.sum(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_avg1", name='PoolSum')
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            tsum[c, h, w] / (kh*kw), \
                            tag='pool_avg2', name='PoolAvg')
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")
    return X, Y, PaddedX


# Defined in file: ./chapter_common_operators/pooling.md
def get_pool_data_mxnet(c: int, n: int, k: int, p: int, s: int, ctx: str='cpu'):
    """Return MXNet data for pooling benchmark using get_conv_data."""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    ctx_obj = getattr(mx, ctx)()
    # get_conv_data is used to generate data where output shape corresponds to pool output.
    data, _, out = get_conv_data(c, c, n, k, p, s, # Use  prefix as in original
                                      constructor=lambda x: mx.nd.array(x, ctx=ctx_obj))
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    return data, out


# Defined in file: ./chapter_common_operators/pooling.md
def pool_mxnet(pool_type: str, data: 'mx.nd.NDArray', out: 'mx.nd.NDArray', k: int, p: int, s: int):
    """Pooling in MXNet"""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    mx.nd.Pooling(data, kernel=(k,k), stride=(s,s),
                      pad=(p,p), pool_type=pool_type, out=out)


# Defined in file: ./chapter_common_operators/batch_norm.md
# from tvm import topi
def batch_norm(c, n, eps=1e-5):
    """batch normalization
    
    c : channels
    N : input width and height
    eps : small positive value to prevent divide 0
    """
        
    X = te.placeholder((c, n, n), name='X')
    Mean = te.placeholder((c, 1, 1), name='Mean')
    Var = te.placeholder((c, 1, 1), name='Var')
    Gamma = te.placeholder((c, 1, 1), name='Gamma')
    Beta = te.placeholder((c, 1, 1), name='Beta')
    C1 = X - Mean
    C2 = topi.sqrt(Var + eps)
    Y = C1 / C2 * Gamma + Beta
    return X, Mean, Var, Gamma, Beta, Y


# Defined in file: ./chapter_common_operators/batch_norm.md
def get_bn_data(c, n, constructor=None):
    """Return the batch norm data, mean, variance, gamma and beta tensors.
       Also return the empty tensor for output.

    c : channels
    n : input width and height
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(c, n, n)).astype('float32')
    mean = np.random.normal(size=(c, 1, 1)).astype('float32')
    # move the mean of the normal distribution to be 1
    var = np.random.normal(loc=1.0, size=(c, 1, 1)).astype('float32')
    # make sure all variance numbers are not negative
    var = np.absolute(var)
    gamma = np.random.normal(size=(c, 1, 1)).astype('float32')
    beta = np.random.normal(size=(c, 1, 1)).astype('float32')
    out = np.empty((c, n, n), dtype='float32')
    if constructor:
        data, mean, var, gamma, beta, out = \
        (constructor(x) for x in [data, mean, var, gamma, beta, out])
    return data, mean, var, gamma, beta, out


# Defined in file: ./chapter_common_operators/batch_norm.md
def get_bn_data_mxnet(c: int, n: int, ctx: str='cpu'):
    """Return MXNet data for batch norm benchmark using get_bn_data."""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    ctx_obj = getattr(mx, ctx)()
    data, mean, var, gamma, beta, out = get_bn_data(c, n, # Use  prefix as in original
                                      lambda x: mx.nd.array(x, ctx=ctx_obj))
    data, out = data.expand_dims(axis=0), out.expand_dims(axis=0)
    return data, mean, var, gamma, beta, out


# Defined in file: ./chapter_common_operators/batch_norm.md
def batch_norm_mxnet(data: 'mx.nd.NDArray', mean: 'mx.nd.NDArray', var: 'mx.nd.NDArray',
                     gamma: 'mx.nd.NDArray', beta: 'mx.nd.NDArray', out: 'mx.nd.NDArray', eps: float=1e-5):
    """Batch normalization in MXNet"""
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    mx.nd.BatchNorm(data, gamma, beta, mean, var, eps,
                    use_global_stats=True, fix_gamma=False, out=out)


# Defined in file: ./chapter_cpu_schedules/call_overhead.md
def bench_workload(workload):
    """Benchmark a workload
    
    workload: a method that accept a num_repeat argument 
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats


# Defined in file: ./chapter_cpu_schedules/vector_add.md
def plot(X, Y, xlabel: str=None, ylabel: str=None, legend: list=None, xlim: tuple=None,
         ylim: tuple=None, xscale: str='linear', yscale: str='linear', fmts: list=None,
         figsize: tuple=(4.5, 3)):
    """Plot multiple lines"""
    # Use set_matplotlib_formats for IPython environments if available
    try:
        ipython_display.set_matplotlib_formats('svg')
    except AttributeError:
        pass # Not in IPython typically

    fig, axes = plt.subplots(figsize=figsize)

    X_arr = np.array(X)
    Y_arr = np.array(Y)

    # Coerce X_arr to a list of arrays, one for each Y series, if X_arr represents shared x-values.
    # This ensures `zip(X_arr_for_plot, Y_data_for_plot, fmts)` works correctly when Y has multiple series.
    if Y_arr.ndim > 1: # Y contains multiple series
        # If X is 1D and its length matches the number of points in each Y series
        if X_arr.ndim == 1 and X_arr.shape[0] == Y_arr.shape[1]:
            X_data_for_plot = [X_arr] * Y_arr.shape[0]
        else: # Otherwise, assume X_arr itself needs to be iterated or matched as is
            X_data_for_plot = [X_arr[i] if X_arr.ndim > 1 else X_arr for i in range(Y_arr.shape[0])]
        Y_data_for_plot = Y_arr
    else: # Y is a single series
        X_data_for_plot = [X_arr]
        Y_data_for_plot = [Y_arr]

    if not fmts: fmts = ['-'] * len(Y_data_for_plot)

    for x_series, y_series, fmt in zip(X_data_for_plot, Y_data_for_plot, fmts):
        axes.plot(x_series, y_series, fmt)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()
    plt.tight_layout()


# Defined in file: ./chapter_cpu_schedules/vector_add.md
def plot_gflops(sizes: np.ndarray, gflops: list, legend: list, xlabel: str='Size'):
    # Use direct call as plot is in the same "module" scope
    plot(sizes, gflops, xlabel=xlabel, ylabel='GFLOPS', # Retain plot prefix
             xscale='log', yscale='log',
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])


# Defined in file: ./chapter_cpu_schedules/vector_add.md
def bench_vector_add_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 0)
        a, b, c = get_abc(n, lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    return sizes / 1e9 / np.array(times)


# Defined in file: ./chapter_cpu_schedules/broadcast_add.md
def bench_bcast_add_tvm(func, sizes: np.ndarray, target: str) -> np.ndarray:
    def workload(nrepeats: int):
        timer = mod.time_evaluator("main", ctx=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats

    times = []
    for n in sizes:
        n_int = int(n)
        s, (A, B, C) = func(n_int)
        mod = tvm.build(s, [A, B, C], target=target)
        ctx = tvm.device(target, 0) # Modern tvm.device usage
        a, b, c = get_bcast_data((n_int, 1), (n_int, n_int), lambda x: tvm.nd.array(x, device=ctx)) # Retain  prefix
        times.append(bench_workload(workload)) # Retain  prefix
    return (sizes * sizes) / 1e9 / np.array(times) # Corrected to use sizes (N) for N*N elements calculation (implicit square output)


# Defined in file: ./chapter_cpu_schedules/matmul.md
def np_matmul_timer(n):
    timer = timeit.Timer(setup='import numpy as np\n'
                         'import d2ltvm\n'
                         'a, b, c = get_abc(%s)' % str((n,n)),
                         stmt = 'np.dot(a, b, out=c)')
    return timer.timeit


# Defined in file: ./chapter_cpu_schedules/matmul.md
def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 0)
        a, b, c = get_abc((n, n), lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    return 2 * sizes**3 / 1e9 / np.array(times)


# Defined in file: ./chapter_cpu_schedules/conv.md
def conv_gflop(oc, ic, n, k, p, s):
    """Compute the #floating point operations in a convolution.

    The arguments are output channels oc, input channels ic, input size n,
    kernel size k, padding p and stride s.
    """
    on = conv_out_size(n, k, p, s)
    return 2 * oc * ic * on * on * k * k / 1e9


# Defined in file: ./chapter_cpu_schedules/conv.md
def conv_timer_mxnet(c, n, k, ctx):
    """Benchmark convolution in MXNet

    c : input, output channels
    n : input width and height
    k : kernel width and height
    """
    timer = timeit.Timer(
        setup='import d2ltvm\n'
        'import mxnet as mx\n'
        'c, n, k, p, s = %d, %d, %d, %d, 1\n'
        'data, weight, bias, out = get_conv_data_mxnet(\n'
        '    c, c, n, k, p, s, "%s")'%(c, n, k, (k-1)//2, ctx),
        stmt='conv_mxnet(data, weight, bias, out, k, p, s);'
        'out.wait_to_read()')
    return timer.timeit


# Defined in file: ./chapter_cpu_schedules/conv.md
def bench_conv_mxnet(sizes, ctx='cpu'):
    """Return the GFLOPS of MXNet convolution"""
    return [conv_gflop(c, c, n, k, (k-1)//2, 1) /
            bench_workload(conv_timer_mxnet(c, n, k, ctx))
            for c, n, k in sizes]


# Defined in file: ./chapter_cpu_schedules/conv.md
def bench_conv_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(x, k, y).mean * nrepeats
    gflops, times = [], []
    for (c, n, k) in sizes:
        args = c, c, n, k, (k-1)//2, 1 # oc, ic, n, k, p, s
        s, (X, K, Y) = func(*args)
        mod = tvm.build(s, [X, K, Y], target)
        ctx = tvm.device(target, 0)
        x, k, y = get_conv_data(
            *args, lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
        gflops.append(conv_gflop(*args))
    return np.array(gflops) / np.array(times)


# Defined in file: ./chapter_cpu_schedules/depthwise_conv.md
def bench_depthwise_conv_tvm(func, sizes: list, target: str) -> np.ndarray:
    def workload(nrepeats: int):
        timer = mod.time_evaluator("main", ctx=ctx, number=nrepeats)
        return timer(x, k, y).mean * nrepeats

    gflops, times = [], []
    for (c, n, k) in sizes:
        # args for depthwise_conv: ic, nh, nw, kh, kw, ph, pw, sh, sw
        args_for_depthwise_conv = (c, n, n, k, k, (k-1)//2, (k-1)//2, 1, 1)
        # func is `depthwise_conv`, returns (X_pl, K_pl, Y_pl, symbolic_padded_input)
        s, tensors = func(*args_for_depthwise_conv)
        X, K, Y = tensors[0], tensors[1], tensors[2]

        mod = tvm.build(s, [X, K, Y], target=target)
        ctx = tvm.device(target, 0)
        # get_conv_data for depthwise: oc, ic_data, n, k, p, s, constructor, conv_type='depthwise'
        x, k_data, y_dummy = get_conv_data(
            c, c, n, k, (k-1)//2, 1, lambda data: tvm.nd.array(data, ctx=ctx), conv_type='depthwise') # Retain  prefix

        times.append(bench_workload(workload)) # Retain  prefix
        # conv_gflop(oc=1, ic=c, n=n, k=k, p=(k-1)//2, s=1)
        gflops.append(conv_gflop(1, c, n, k, (k-1)//2, 1)) # Retain  prefix
    return np.array(gflops) / np.array(times)


# Defined in file: ./chapter_cpu_schedules/depthwise_conv.md
def depthwise_conv_timer_mxnet(c: int, n: int, k: int, ctx: str):
    """Benchmark depthwise convolution in MXNet

    c : input, output channels
    n : input width and height
    k : kernel width and height
    """
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    return timeit.Timer(
        setup='import mxnet as mx\n'
              'import d2ltvm\n'
              'c, n, k, p, s = %d, %d, %d, %d, 1\n'
              'data, weight, bias, out = get_conv_data_mxnet(\n'
              '    c, c, n, k, p, s, "%s", "%s")' % (c, n, k, (k-1)//2, ctx, 'depthwise'),
        stmt='depthwise_conv_mxnet(data, weight, bias, out, k, p, s);'
             'out.wait_to_read()')


# Defined in file: ./chapter_cpu_schedules/depthwise_conv.md
def bench_depthwise_conv_mxnet(sizes: list, ctx: str='cpu') -> np.ndarray:
    """Return the GFLOPS of MXNet depthwise convolution"""
    return np.array([conv_gflop(1, c, n, k, (k-1)//2, 1) / # Retain  prefix
            bench_workload(depthwise_conv_timer_mxnet(c, n, k, ctx)) # Retain  prefix
            for c, n, k in sizes])


# Defined in file: ./chapter_cpu_schedules/pooling.md
def bench_pooling_tvm(func, sizes: list, target: str) -> np.ndarray:
    """Benchmark pooling in TVM

    func : the scheduling method, e.g. `schedule_pool_max_cpu`
           Expected to return (schedule, [input_tensor, output_tensor, optional_padded_tensor])
    sizes : the data size list, each of which is a (channel, input_hw, kernel_hw) triplet
    target : the TVM target, e.g. llvm or cuda
    """
    def workload(nrepeats: int):
        timer = mod.time_evaluator("main", ctx=ctx, number=nrepeats)
        # Note: `data` and `out_max` are created outside this lambda's scope.
        return timer(data, out_max).mean * nrepeats

    times = []
    for size in sizes:
        c, n, k = size
        # func expects (pool_type, c, nh, nw, kh, kw, ph, pw, sh, sw)
        # Use 'max' pool_type as a default for benchmarking in this utility.
        # The chapter itself defines how to call func for max/avg.
        # Original bench_pooling_tvm also used 'out_max' (for max pool) in workload.
        # So we infer this func is likely for max pool as per original context.
        func_args = ('max', c, n, n, k, k, 1, 1, 1, 1) # Example for max pooling, p=1, s=1
        sch, tensors = func(*func_args) # func is `pool`, returns (X_pl, Y_pl, symbolic_padded_input)
        X, Y = tensors[0], tensors[1] # Actual inputs to build are X, Y

        mod = tvm.build(sch, [X, Y], target=target)
        ctx = tvm.device(target, 0)
        # get_conv_data is used for data generation, where `oc` and `ic` are `c`.
        # `n` is `n_actual`, `k` `k_actual`, `p` and `s` are hardcoded `1`.
        data_np, _, _ = get_conv_data(c, c, n, k, 1, 1) # Retain  prefix
        data = tvm.nd.array(data_np, ctx=ctx)

        # Create empty output NDArray matching Y's shape
        oh = conv_out_size(n, k, 1, 1) # Retain  prefix
        ow = conv_out_size(n, k, 1, 1)
        out_max = tvm.nd.array(np.empty((c, oh, ow), dtype='float32'), ctx=ctx)

        times.append(bench_workload(workload)) # Retain  prefix
    return np.array(times)


# Defined in file: ./chapter_cpu_schedules/pooling.md
def pooling_timer_mxnet(pool_type: str, c: int, n: int, k: int, ctx: str):
    """Benchmark pooling in MXNet

    c : channels
    n : input width and height
    k : kernel width and height
    """
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    return timeit.Timer(
        setup='import mxnet as mx\n'
              'import d2ltvm\n'
              'c, n, k, p, s = %d, %d, %d, 1, 1\n'
              'data, out = get_pool_data_mxnet(\n'
              '    c, n, k, p, s, "%s")' % (c, n, k, ctx),
        stmt='pool_mxnet("%s", data, out, k, p, s);'
             'out.wait_to_read()' % (pool_type))


# Defined in file: ./chapter_cpu_schedules/pooling.md
def bench_pooling_mxnet(pool_type: str, sizes: list, ctx: str='cpu') -> np.ndarray:
    """Return the execution times of MXNet pooling"""
    return np.array([bench_workload(pooling_timer_mxnet(pool_type, c, n, k, ctx)) # Retain  prefix
            for c, n, k in sizes])


# Defined in file: ./chapter_cpu_schedules/batch_norm.md
def bench_bn_tvm(func, sizes, target):
    """Benchmark batch normalization in TVM

    func : the scheduling method
    sizes : the data size list, each of which is a (channel, input_hw) tuple
    target : the TVM target, e.g. llvm or cuda
    """
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(data, mean, var, gamma, beta, out).mean * nrepeats
    times = []
    for size in sizes:
        sch, args = func(size)
        mod = tvm.build(sch, args, target)
        ctx = tvm.device(target, 0)
        # data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1], 
        #                                                        lambda x: tvm.nd.array(x, device=ctx))
        data, mean, var, gamma, beta, out = get_bn_data(size[0], size[1], 
                                                               lambda x: tvm.nd.array(x, device=ctx))
        # times.append(bench_workload(workload))
        times.append(bench_workload(workload))
    return np.array(times)


# Defined in file: ./chapter_cpu_schedules/batch_norm.md
def bn_timer_mxnet(c: int, n: int, ctx: str):
    """Benchmark batch normalization in MXNet

    c : channels
    n : input width and height
    ctx : compute ctx, e.g., cpu or gpu
    """
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    return timeit.Timer(
        setup='import mxnet as mx\n'
              'import d2ltvm\n'
              'c, n = %d, %d\n'
              'data, mean, var, gamma, beta, out = get_bn_data_mxnet(\n'
              '    c, n, "%s")' % (c, n, ctx),
        stmt='batch_norm_mxnet(data, mean, var, gamma, beta, out);'
             'out.wait_to_read()')


# Defined in file: ./chapter_cpu_schedules/batch_norm.md
def bench_bn_mxnet(sizes: list, ctx: str='cpu') -> np.ndarray:
    """Return the execution times of MXNet batch norm"""
    # return np.array([bench_workload(bn_timer_mxnet(c, n, ctx)) # Retain  prefix
    #         for c, n in sizes])
    return np.array([bench_workload(bn_timer_mxnet(c, n, ctx)) # Retain  prefix
            for c, n in sizes])


# Defined in file: ./chapter_gpu_schedules/matmul.md
def matmul_timer_mxnet(n: int, ctx: str):
    """The matrix multiplication timer for MXNet

    n : width and height of inputs
    ctx : device
    """
    # if 'mx' not in sys.modules:
    #     raise ImportError("MXNet not imported. Please install MXNet to use this function.")
    return timeit.Timer(
        setup='import mxnet as mx\n'
              'import d2ltvm\n'
              'a, b, c, = get_abc((%d, %d), lambda x: mx.nd.array(x, ctx=mx.%s()))\n'
              'mx.nd.waitall()' % (n, n, ctx),
        stmt='mx.nd.dot(a, b, out=c); c.wait_to_read()')


# Defined in file: ./chapter_gpu_schedules/conv.md
def split_axis(factors, sch, op, axis):
        """Splitting an axis into factors

        Parameters
        ----------
        factors: array of integers
            The factors that the split applies
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        op: tvm.te.tensor.Operation
            The stage to be applied
        axis: tvm.te.schedule.IterVar
            axis to split

        Returns
        -------
        axes : list of Axis
            The transformed axes.
        """
        ret = []
        for i in range(0, len(factors)):
            ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]