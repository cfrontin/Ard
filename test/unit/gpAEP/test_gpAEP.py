import numpy as np

import gpAEP

import pytest


class Test_ProductionToolBase:

    def setup_method(self):

        # setup a small dummy resource function

        # create a set of independent variables
        Nx = 16
        Ny = 25
        Nz = 20
        X, Y, Z = self.X, self.Y, self.Z = np.meshgrid(
            *[np.linspace(0.0, 1.0, Nd) for Nd in [Nx, Ny, Nz]], indexing="ij"
        )

        # create and store test function and alternate plus evals
        self.potential_function = lambda x, y, z: np.exp(
            -(0.25 * x**2 + 0.125 * y**2 + 0.5 * z**2)
        )
        self.response_function = lambda x, y, z: 1.0 - (
            0.75 * x**2 + 0.25 * y**2 + 0.25 * z**2
        )
        self.potential_val = self.potential_function(X, Y, Z)
        self.response_val = self.response_function(X, Y, Z)

        # create a base production tool
        self.ptb = gpAEP.gpAEP.ProductionToolBase(self.potential_val)

        self.integral_exact = 0.4643685883607919

    def test_indep(self):
        """make sure the independent variable handling is working"""

        # loop over the dimensions and make sure the normalized independent
        # variables are right
        ptb_indep = self.ptb.get_indep_dim()  # get the list version
        assert (
            len(ptb_indep) == self.potential_val.ndim
        )  # should have the right number of dimensions
        assert (
            self.ptb.get_ndim() == self.potential_val.ndim
        )  # should think it has the right number of dimensions
        for d in range(self.potential_val.ndim):
            ptb_indep_d = self.ptb.get_indep_dim(d)  # get the single dimension version

            # make sure both the overall and dimension-wise versions match
            assert np.all(
                np.isclose(
                    ptb_indep[d], np.linspace(0.0, 1.0, self.potential_val.shape[d])
                )
            )
            assert np.all(
                np.isclose(
                    ptb_indep_d, np.linspace(0.0, 1.0, self.potential_val.shape[d])
                )
            )

        # make sure an exception occurs if we have ask for a nonexistent dimension
        with pytest.raises(IndexError):
            self.ptb.get_indep_dim(self.potential_val.ndim)

    def test_potential_values(self):
        """make sure the potential storage is working"""

        # make sure the potential values called by the solver are
        assert np.allclose(self.ptb.get_potential_values(), self.potential_val)

    def test_response_function(self):
        """make sure the response function handling is working"""

        # get the independent variable locations
        X, Y, Z = self.ptb.get_indep_mesh()

        # make sure the response function (currently not stored) is None
        assert self.ptb.response_function is None

        # make sure if we set in a response function is gets the right answer
        self.ptb.set_response_function(self.response_function)
        assert self.ptb.response_function is not None  # shouldn't be None anymore
        assert np.allclose(self.ptb.response_function(X, Y, Z), self.response_val)

    def test_production_integrand_function(self):
        """make sure the production integrand function computation is working"""

        # make sure the response function (currently not stored) is None
        assert self.ptb.response_function is None
        with pytest.raises(AssertionError):
            self.ptb.eval_production_integrand_exact()

        # make sure if we set in a response function is gets the right answer
        self.ptb.set_response_function(self.response_function)
        assert self.ptb.response_function is not None  # shouldn't be None anymore

        # now evaluate the integrand and validate its correctness
        assert np.allclose(
            self.ptb.eval_production_integrand_exact(),
            self.response_val * self.potential_val,
        )

    def test_production_integral_value(self):
        """make sure the production integral computation is working"""

        # make sure the response function (currently not stored) is None
        assert self.ptb.response_function is None
        with pytest.raises(AssertionError):
            self.ptb.get_production_exact()

        # make sure if we set in a response function is gets the right answer
        self.ptb.set_response_function(self.response_function)
        assert self.ptb.response_function is not None  # shouldn't be None anymore

        # now evaluate the integrand and validate its correctness
        assert np.isclose(
            self.ptb.get_production_exact(),
            self.integral_exact,
            rtol=5.0e-3,  # loose tolerance, not a ton of points
        )


class Test_gpProductionToolBase(Test_ProductionToolBase):

    def setup_method(self):

        # setup identically to base class version
        super().setup_method()

        # extract independent variables
        X, Y, Z = self.X, self.Y, self.Z

        # overwrite with a GP production tool
        self.ptb = gpAEP.gpAEP.gpProductionToolBase(self.potential_val)

    def test_super_indep(self):
        """pass through to super test to ensure inheritence is working"""
        super().test_indep()

    def test_super_potential_values(self):
        """pass through to super test to ensure inheritence is working"""
        super().test_potential_values()

    def test_super_response_function(self):
        """pass through to super test to ensure inheritence is working"""
        super().test_response_function()

    def test_super_production_integrand_function(self):
        """pass through to super test to ensure inheritence is working"""
        super().test_production_integrand_function()

    def test_super_production_integral_value(self):
        """pass through to super test to ensure inheritence is working"""
        super().test_production_integral_value()

    def test_setup_kernel(self):
        """make sure kernel setup works"""

        # create a training set based on the true function
        Nt = 20
        XYZt = Xt, Yt, Zt = [np.random.rand(Nt) for d in range(self.ptb.get_ndim())]
        Ft = self.response_function(Xt, Yt, Zt)
        self.ptb.setup_kernel(XYZt, Ft, optuna_timeout=0.25)

        # make sure a GP kernel was created and has a nonzero set of parameters
        assert self.ptb.kernel is not None
        assert np.all(self.ptb.integrandGP.kernel_.theta != 0.0)


class Test_gpProductionIntegrator(Test_gpProductionToolBase):

    def setup_method(self):

        # setup identically to base class version
        super().setup_method()

        # extract independent variables
        X, Y, Z = XYZ = [self.X, self.Y, self.Z]

        # overwrite with a GP production tool
        self.integ = gpAEP.gpProductionIntegrator(self.potential_val)

    def test_grandsuper_indep(self):
        """pass through to grand-super test to ensure inheritence is working"""
        super().test_super_indep()

    def test_grandsuper_potential_values(self):
        """pass through to grand-super test to ensure inheritence is working"""
        super().test_super_potential_values()

    def test_grandsuper_response_function(self):
        """pass through to grand-super test to ensure inheritence is working"""
        super().test_super_response_function()

    def test_grandsuper_production_integrand_function(self):
        """pass through to grand-super test to ensure inheritence is working"""
        super().test_super_production_integrand_function()

    def test_grandsuper_production_integral_value(self):
        """pass through to grand-super test to ensure inheritence is working"""
        super().test_super_production_integral_value()
