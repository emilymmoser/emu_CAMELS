import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
from scipy.interpolate import Rbf
import ostrich.mathutils


class GaussianProcessInterpolator:
    def __init__(
            self,
            params,
            func_vals,
            kernel=sklearn.gaussian_process.kernels.RBF(),
    ):
        params = ostrich.mathutils.atleast_kd(params, 2)
        self.gpi = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(params, func_vals)

    def __call__(self, params):
        params = ostrich.mathutils.atleast_kd(params, 2)
        return self.gpi.predict(params)


"""
class RbfInterpolator:
    rbf_function = 'multiquadric'

    def __init__(self, params, func_vals):
        params = ostrich.mathutils.atleast_kd(params, 2)
        point_vals = func_vals.flatten()
        self.rbfi = Rbf(*params.T, point_vals, function=self.rbf_function)

    def __call__(self, params):
        params = ostrich.mathutils.atleast_kd(params, 2)
        return self.rbfi(*params.T)
"""
class RbfInterpolator:
  def __init__(self, params, func_vals, function='multiquadric'):
    params = ostrich.mathutils.atleast_kd(params, 2)
    point_vals = func_vals.flatten()
    self.rbfi = Rbf(*params.T, point_vals, function=function)
  def __call__(self, params):
    params = ostrich.mathutils.atleast_kd(params, 2)
    return self.rbfi(*params.T)