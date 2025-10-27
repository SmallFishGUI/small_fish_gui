import inspect
import datetime as dt

def check_parameter(**kwargs):
    """Check dtype of the function's parameters.

    Parameters
    ----------
    kwargs : Type or Tuple[Type]
        Map of each parameter with its expected dtype.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # get the frame and the parameters of the function
    frame = inspect.currentframe().f_back
    _, _, _, values = inspect.getargvalues(frame)

    # compare each parameter with its expected dtype
    for arg in kwargs:
        expected_dtype = kwargs[arg]
        parameter = values[arg]
        if not isinstance(parameter, expected_dtype):
            actual = "'{0}'".format(type(parameter).__name__)
            if isinstance(expected_dtype, tuple):
                target = ["'{0}'".format(x.__name__) for x in expected_dtype]
                target = "(" + ", ".join(target) + ")"
            else:
                target = expected_dtype.__name__
            raise TypeError("Parameter {0} should be a {1}. It is a {2} "
                            "instead.".format(arg, target, actual))

    return True


def compute_anisotropy_coef(voxel_size) :
    """
    Returns tuple (anisotropy_z, anisotropy_y, 1)
    voxel_size : tuple (z,y,x).
    """

    if not isinstance(voxel_size, (tuple, list)) : raise TypeError("Expected voxel_size tuple or list")
    if len(voxel_size) == 2 : is_3D = False
    elif len(voxel_size) == 3 : is_3D = True
    else : raise ValueError("Expected 2D or 3D voxel, {0} element(s) found".format(len(voxel_size)))

    if is_3D :
        z_anisotropy = voxel_size[0] / voxel_size [2]
        xy_anisotropy = voxel_size[1] / voxel_size [2]
        return (z_anisotropy, xy_anisotropy, 1)

    else :
        return (voxel_size[0] / voxel_size[1], 1)
    

def get_datetime():
    return dt.datetime.now().strftime("%Y%m%d %H-%M-%S")