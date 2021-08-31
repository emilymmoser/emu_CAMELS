import pytest
from pretend import stub
import numpy as np
import ostrich.interpolate


def describe_GaussianProcessInterpolator():

    def describe__call__():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)
            func_vals = np.ones(params.shape)
            interpolator = ostrich.interpolate.GaussianProcessInterpolator(params, func_vals)
            params_to_eval = np.linspace(0, 1, 20)
            result = interpolator(params_to_eval)
            assert np.allclose(result, 1)


def describe_RbfInterpolator():

    def describe__call__():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)
            func_vals = np.ones(10)
            interpolator = ostrich.interpolate.RbfInterpolator(params, func_vals)
            params_to_eval = np.linspace(0, 1, 20)
            result = interpolator(params_to_eval)
            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            p = np.arange(1, 5)
            params = np.stack((p, p, p, p, p, p)).T
            func_vals = np.ones(4)
            interpolator = ostrich.interpolate.RbfInterpolator(params, func_vals)
