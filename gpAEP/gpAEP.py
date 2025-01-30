import typing
import abc

import numpy as np
from numpy.core.multiarray import array as array
import numpy.random as rng
import scipy.optimize as opt
from scipy.special import gamma

# from scipy.spatial.distance import cdist, pdist, squareform

from sklearn.gaussian_process import GaussianProcessRegressor as skl_gpr
from sklearn.gaussian_process.kernels import Matern as skl_matern

# from sklearn.gaussian_process.kernels import ExpSineSquared as skl_periodic
# from sklearn.gaussian_process.kernels import Product as skl_product

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# from gpAEP.kernel import kernel_matern52, dkernel_dXYZ0_matern52


class ProductionToolBase(object):
    """
    A base class for a assessment-efficient production estimation tool.

    This tool will take a Nd-dimensional description of a resource potential,
    mathematically $p(x_1, \\ldots, x_d)$ and represented by a grid
    `potential_bins`

    Parameters
    ----------
    potential_bins : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    def __init__(
        self,
        potential_bins: np.array,
        response_function: typing.Callable[..., float] = None,
    ) -> None:

        # store the defining grids of the independent variables
        self.indep_bins = [
            np.linspace(0.0, 1.0, len_d) for len_d in potential_bins.shape
        ]
        self.potential_bins = np.copy(potential_bins)

        # store the response function
        self.response_function = response_function

        # set up default values
        self._default_matern_lengthscales = [1.0 / 10 for d in range(self.get_ndim())]
        self._default_matern_lengthscale_bounds = [
            [1.0e-6, 1e3] for d in range(self.get_ndim())
        ]

    def get_ndim(self) -> int:
        """get the number of dimensions on which the potential is defined"""
        return self.potential_bins.ndim

    def get_indep_dim(
        self,
        dimension: int = None,
    ) -> np.array:
        """
        get the locations that define the independent variable for a given
        dimension (or, if no dimension is given, return a list of all of them)
        """
        if dimension is None:
            return self.indep_bins
        elif dimension < self.get_ndim():
            return self.indep_bins[dimension]
        else:
            raise IndexError(
                f"the provided data only has {len(self.potential_bins)} "
                + f"and you asked for index {dimension}."
            )

    def get_indep_mesh(
        self,
        dimension_list: list[int] = None,
    ) -> list[np.array]:
        """
        return the meshgrids across the list of dimensions requested, or all by
        default
        """

        # by default, meshgrid across all dimensions
        if dimension_list is None:
            dimension_list = list(range(self.get_ndim()))
        mesh_list = np.meshgrid(
            *[self.get_indep_dim(d) for d in dimension_list],
            indexing="ij",
        )
        return mesh_list

    def get_potential_values(self) -> np.array:
        """
        return the potential values as a Nx1 x ... x Nxd array
        """
        return self.potential_bins

    def set_response_function(
        self,
        rf_in: typing.Callable[..., float],
    ):
        """set (or change) an exact farm power curve function"""
        self.response_function = rf_in

    def eval_production_integrand_exact(self) -> np.array:
        """
        return the AEP integrand using an exact response function at the stored points
        """

        assert (
            self.response_function is not None
        ), "exact power function must be supplied to use eval_production_integrand_exact"

        # get the independent variable location values
        XYZ = self.get_indep_mesh()

        # return the potential/response product function
        return self.response_function(*XYZ) * self.get_potential_values()

    def get_production_exact(self) -> float:
        """return the integrated production using an exact response function"""
        assert (
            self.response_function is not None
        ), "exact power function must be supplied to use get_AEP_pf"
        integrand_working = np.copy(self.eval_production_integrand_exact())
        for d in reversed(list(range(self.get_ndim()))):
            integrand_working = np.trapz(
                integrand_working,
                x=self.get_indep_dim(d),
                axis=d,
            )
        return float(integrand_working)


class gpProductionToolBase(ProductionToolBase):

    def __init__(
        self,
        potential_bins: np.array,
        response_function: typing.Callable[..., float] = None,
    ) -> None:

        # initialize AEP meta data
        super().__init__(
            potential_bins,
            response_function,
        )

        # data for GPs
        self.indep_training = None  # make a place to have training data, for later
        self.response_training = None  # make a place to have training data, for later
        self.kernel = None  # make a place to have a kernel, for later
        self.integrandGP = None  # make a place to have a integrand GP, for later

    def setup_kernel(
        self,
        indep_training,  # independent variable training data
        response_training,  # response variable training data
        kernel=None,  # pre-specify a kernel, otherwise it'll be Matern 5/2
        train_lengthscales=True,  # do we want to train the lengthscales just GP hyperparams
        lengthscales=None,  # lengthscales for GP (initial)
        lengthscale_bounds=None,  # bounds for lengthscale optimization
        wrap_kernel=False,  # should we try the wraparound kernel? (i.e. distance [-pi, pi])
        optuna_timeout=10,
    ) -> None:
        """
        set up a GP kernel for a production integral

        feed it a list of vectors of independent variable data, and it
        configures and fits the integrator's GP to be able to make an production
        estimate

        response data is expected to be input in a normalized unit. output from
        the model will be likewise normalized.

        parameters
        ----------
        indep_training : list[np.ndarray]
            independent variable data for training, each list entry should be
            the same size as the others
        response_training : np.ndarray
            response data for training, should be same size as the independent
            variable columns
        kernel : TODO
        train_lengthscales : TODO
        lengthscales : TODO
        lengthscale_bounds : TODO
        wrap_kernel : TODO
        """

        # use fixed lengthscales if specified
        if not train_lengthscales:
            lengthscale_bounds = "fixed"

        if lengthscales is None:
            lengthscales = self._default_matern_lengthscales
        if lengthscale_bounds is None:
            lengthscale_bounds = self._default_matern_lengthscale_bounds

        # save off copied training data for later use
        self.indep_training = [v.copy() for v in indep_training]
        self.response_training = response_training.copy()

        if kernel is None:

            # create the kernel for GP regression
            matern_nu = 2.5
            kernel_matern = 1.0 * skl_matern(
                length_scale=lengthscales,
                length_scale_bounds=lengthscale_bounds,
                nu=matern_nu,
            )
            self.kernel = kernel_matern

            if wrap_kernel:
                raise NotImplementedError("not implemented yet! -cfrontin")

                def kernel_wraparound(XYZ0, XYZ1=None):
                    if XYZ1 is None:
                        XYZ1 = XYZ0.copy()
                    dX = np.abs(np.subtract.outer(XYZ0[:, 0], XYZ1[:, 0]))
                    dY = np.abs(np.subtract.outer(XYZ0[:, 1], XYZ1[:, 1]))
                    dX = np.where(
                        dX < np.pi, dX, 2 * np.pi - dX
                    )  # wrap the differences
                    dD = np.concatenate(dX.flat, dY.flat, axis=-1)
                    # d0 = abs(XYZ0[0] - XYZ1[:, 0]).reshape((-1, 1))
                    # d1 = abs(XYZ0[1] - XYZ1[:, 1]).reshape((-1, 1))
                    # d0 = np.where(d0 < np.pi, d0, 2 * np.pi - d0)
                    # dd = np.concatenate((d0, d1), axis=-1)
                    return kernel_matern([0.0, 0.0], dD).reshape(
                        [XYZ0[:, 0], XYZ0[:, 1]]
                    )

            use_optuna = False
            if use_optuna:
                # create a GP regressor
                self.integrandGP = skl_gpr(
                    kernel=self.kernel,
                    optimizer=None,
                    # n_restarts_optimizer=4,
                )

                # fit the GP with no optimizer... so don't actually fit it
                self.integrandGP.fit(
                    np.vstack(indep_training).T,
                    response_training,
                )

                # create an optuna study to optimize the hyperparameters
                def objective(trial: optuna.Trial) -> float:
                    k1_constant_value = trial.suggest_float(
                        "k1_constant_value", 1.0e-4, 1.0e-1
                    )
                    k2_length_scale = np.zeros((self.get_ndim()))
                    for idx in range(len(k2_length_scale)):
                        k2_length_scale[idx] = trial.suggest_float(
                            f"k2_length_scale_{idx:03d}", 1.0e-4, 1.0e-1
                        )
                    self.integrandGP.set_params(
                        kernel__k1__constant_value=k1_constant_value,
                        kernel__k2__length_scale=k2_length_scale,
                    )
                    lml = self.integrandGP.log_marginal_likelihood()
                    return lml

                study = optuna.create_study(direction="maximize")  # maximize
                study.optimize(
                    objective, timeout=optuna_timeout
                )  # best answer in # of seconds
                self.integrandGP.kernel_.theta = np.array(
                    list(study.best_params.values())
                )  # pack in the result

                # put a reference to the learned kernel in the integrator tool (for later use)
                self.kernel = self.integrandGP.kernel = self.integrandGP.kernel_

                # I DONT THINK THE BELOW IS NECESSARY

                # # fit the GP with no optimizer to guarantee valid values
                # self.integrandGP.fit(
                #     np.vstack(indep_training).T,
                #     response_training,
                # )
                #
                # print(f"DEBUG!!!!! final kernel_.theta: {self.integrandGP.kernel_.theta}")

            else:
                # create a GP regressor
                self.integrandGP = skl_gpr(
                    kernel=self.kernel,
                    # optimizer=None,
                    n_restarts_optimizer=4,
                )

                # fit the GP with no optimizer... so don't actually fit it
                self.integrandGP.fit(
                    np.vstack(indep_training).T,
                    response_training,
                )

                # # create an optuna study to optimize the hyperparameters
                # def objective(trial: optuna.Trial) -> float:
                #     theta = np.zeros_like(self.integrandGP.kernel_.theta)
                #     for idx in range(len(theta)):
                #         theta[idx] = trial.suggest_float(
                #             f"theta_{idx:03d}", 1e-4, 1e1, log=True
                #         )
                #     lml = self.integrandGP.log_marginal_likelihood(theta)
                #     return lml
                #
                # study = optuna.create_study(direction="maximize")  # maximize
                # study.optimize(
                #     objective, timeout=optuna_timeout
                # )  # best answer in # of seconds
                # self.integrandGP.kernel_.theta = np.array(
                #     list(study.best_params.values())
                # )  # pack in the result

                # put a reference to the learned kernel in the integrator tool (for later use)
                self.kernel = self.integrandGP.kernel = self.integrandGP.kernel_

        else:
            self.kernel = kernel

    def _get_Kcov(
        self,
        indep_eval=None,
    ) -> np.array:
        """get the covariance kernel on the training data"""

        # by default, get kstar at the training points
        if indep_eval is not None:
            indep_eval = self.indep_training

        # get the covariance matrix of evaluation points
        Kcov = self.kernel(
            np.vstack([v.flat for v in indep_eval]).T
        )

        return Kcov

    def _get_kstar(
        self,
        indep_query: typing.Union[list[np.array], list[float]],
        indep_ref=None,
    ) -> np.array:
        """get a covariance matrix of the training data w.r.t. a query"""

        # by default, get kstar at the training points
        if indep_ref is None:
            indep_ref = self.indep_training

        # roll up inputs and evaluate the kernel
        XYZq = np.vstack([v.flat for v in indep_query]).T
        XYZr = np.vstack([v.flat for v in indep_ref]).T
        kstar = self.kernel(XYZq, XYZr)

        return kstar

    def _get_w(
        self,
        indep_training: typing.Union[None, list[np.array], list[float]] = None,
    ) -> np.array:
        """get the w vector from King paper"""

        # by default, get kstar at the training points
        if indep_training is None:
            indep_training = self.indep_training

        # by default, evaluate at the pmf points
        indep_data = self.get_indep_mesh()

        # evaluate kstar first
        kstar = self._get_kstar(
            indep_data,
            indep_ref=indep_training,
        )

        Ndim = self.get_ndim()  # number of dimensions
        dim_shapes = self.get_potential_values().shape  # dimension sizes

        indices_available = "ijklmnopq"
        if len(indices_available) < (Ndim + 3):
            raise NotImplementedError(
                f"did not implement GP for more than "
                + f"{len(indices_available) - 3} dimensions"
            )

        # create an integrand using the einstein summation
        ein0 = indices_available[:Ndim]
        ein1 = indices_available[:(Ndim+1)]
        integrand_local = np.einsum(
            f"{ein0},{ein1}->{ein1}",
            self.get_potential_values(),
            kstar.reshape(*dim_shapes, -1),  # expand front-end dims of the k* for integration
        )

        # apply approx integration via pdf-weighted sum
        for d in reversed(range(Ndim)):
            integrand_local = np.trapz(
                integrand_local, x=self.get_indep_dim(d), axis=d,
            )

        print(f"DEBUG!!!!! integrand_local shape: {integrand_local.shape}")
        w_AEP = integrand_local

        return w_AEP

    def _get_wTinvK(
        self,
        indep_training: typing.Union[list[np.array], list[float]] = None,
    ) -> np.array:
        """get w^T*inv(K) from King paper (useful for many reasons)"""

        # by default, get kstar at the training points
        if indep_training is None:
            indep_training = self.indep_training

        # by default, evaluate at the PMF points
        indep_eval = self.get_indep_mesh()

        # grab the terms
        w_AEP = self._get_w(
            indep_training,
        )
        Kcov = self._get_Kcov(
            indep_training,
        )

        # combine and ship
        wTinvK_AEP = w_AEP @ np.linalg.inv(Kcov)
        return wTinvK_AEP

    def _get_wTinvKw(
        self,
        indep_training: typing.Union[list[np.array], list[float]] = None,
        scale=1.0,
    ) -> np.array:
        """get w^T*inv(K) from King paper (useful for many reasons)"""

        # by default, get kstar at the training points
        if indep_training is None:
            indep_training = self.indep_training

        # # by default, evaluate at the PMF points
        # indep_eval = self.get_indep_mesh()

        w_AEP = self._get_w(indep_training)
        Kcov = self._get_Kcov(indep_training)

        wTinvKw_AEP = w_AEP @ np.linalg.solve(Kcov, w_AEP)
        return scale * wTinvKw_AEP

    def _get_Z(
        self,
        indep_ref: typing.Union[list[np.array], list[float]] = None,
    ) -> np.array:

        if indep_ref is None:
            # by default, evaluate at the PMF points
            indep_ref = self.get_indep_mesh()

        """get the Z term from the variance from King paper"""
        XYZq = np.vstack([v.flat for v in indep_ref]).T
        Zval = np.sum(
            self.get_potential_values().reshape(-1)
            @ self.kernel(XYZq)
            @ (self.get_potential_values().reshape(-1))
        )
        return Zval

    # def _get_matern52_rk(
    #     self,
    #     XYZ0,
    #     XYZ1,
    #     lengthscale=ProductionToolBase._default_matern_lengthscales,
    #     wrap_kernel=False,
    # ):
    #     XYZ0 = np.atleast_2d(XYZ0)
    #     XYZ1 = np.atleast_2d(XYZ1)
    #
    #     k = skl_matern(
    #         length_scale=lengthscale,
    #         length_scale_bounds="fixed",
    #         nu=2.5,
    #     )
    #     # def k(XYZ0, XYZ1):
    #     #     dists = cdist(XYZ0 / lengthscale, XYZ1 / lengthscale, metric="euclidean")
    #     #     K = dists * np.sqrt(5)
    #     #     K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
    #     #     if XYZ1 is None:
    #     #         # convert from upper-triangular matrix to square matrix
    #     #         K = squareform(K)
    #     #         np.fill_diagonal(K, 1)
    #     #     return K
    #
    #     if wrap_kernel:
    #         d0 = abs(np.subtract.outer(XYZ0[:, 0], XYZ1[:, 0])).reshape((-1, 1))
    #         d1 = abs(np.subtract.outer(XYZ0[:, 1] - XYZ1[:, 1])).reshape((-1, 1))
    #         d0 = np.where(d0 < np.pi, d0, 2 * np.pi - d0)
    #         dd = np.concatenate((d0, d1), axis=-1)
    #
    #         k_out = k([0, 0], dd).reshape([XYZ0.shape[0], XYZ1.shape[0]])
    #     else:
    #         k_out = k(XYZ0, XYZ1)
    #
    #     return k_out

    # def _get_dmatern52_rk(
    #     self,
    #     XYZ0,
    #     XYZ1,
    #     lengthscale=ProductionToolBase._default_matern_lengthscales,
    #     wrap_kernel=False,
    # ) -> np.array:
    #
    #     def dk_i(XYZ0, XYZ1):
    #         dK = np.zeros((2, 2))
    #         d = np.sqrt((XYZ1 - XYZ0) ** 2)
    #         dx0_d = -1 / d * (XYZ1[0] - XYZ0[0])
    #         dy0_d = -1 / d * (XYZ1[1] - XYZ0[1])
    #         dx1_d = 1 / d * (XYZ1[0] - XYZ0[0])
    #         dy1_d = 1 / d * (XYZ1[1] - XYZ0[1])
    #         K0 = d * np.sqrt(5)
    #         dd_K0 = np.sqrt(d)
    #         K = (1.0 + K0 + K0**2 / 3.0) * np.exp(-K0)
    #         dK_dK0 = (dd_K0 + 2 * K0 * dd_K0 / 3.0) * np.exp(-K0) + -K0 * (
    #             1.0 + K0 + K0**2 / 3.0
    #         ) * np.exp(-K0)
    #
    #         dK = dd_K0 @ dK_dK0
    #
    #         return K, dK_dK0
    #
    #     dk = np.zeros((XYZ0.shape[0], XYZ1.shape[0], 2))
    #
    #     for k in range(XYZ0.shape[0]):
    #
    #         d0 = abs(XYZ0[k, 0] - XYZ1[:, 0]).reshape((-1, 1))
    #         d1 = abs(XYZ0[k, 1] - XYZ1[:, 1]).reshape((-1, 1))
    #         if wrap_kernel:
    #             d0 = np.where(d0 < np.pi, d0, 2 * np.pi - d0)
    #         dd = np.concatenate((d0, d1), axis=-1)
    #
    #         r = np.sqrt(np.sum((dd / lengthscale) ** 2, axis=1))
    #
    #         dk[k, :, :] = (
    #             -(5.0 / 3.0)
    #             * np.sqrt(r)
    #             * (1 + np.sqrt(5) * r)
    #             * np.exp(-np.sqrt(5) * r)
    #         ).reshape((-1, 1)) * (dd / lengthscale)
    #
    #     return dk

    # def _get_dkstar(
    #     self,
    #     direction_query: typing.Union[np.array, float],
    #     speed_query: typing.Union[np.array, float],
    #     direction_ref=None,
    #     speed_ref=None,
    # ) -> np.array:
    #     """get a covariance matrix of the training data w.r.t. a query"""
    #
    #     raise NotImplementedError("this hasn't been implemented yet. -cfrontin")
    #
    #     # by default, get kstar at the training (pmf) points
    #     if direction_ref is None:
    #         direction_ref = self.indep_training
    #     if speed_ref is None:
    #         speed_ref = self.speed_training
    #
    #     XYZq = np.vstack([direction_query.flat, speed_query.flat]).T
    #     XYZt = np.vstack([direction_ref.flat, speed_ref.flat]).T
    #     dkstar_dX, dkstar_dY = dkernel_dXYZ0_matern52(
    #         XYZt,
    #         XYZq,
    #         lengthscale=self.kernel.get_params()["k2__length_scale"],
    #     )
    #     dkstar_dX = [v.T for v in dkstar_dX]
    #     dkstar_dY = [v.T for v in dkstar_dY]
    #     return dkstar_dX, dkstar_dY

    # def _get_dw(
    #     self,
    #     indep_training: typing.Union[None, list[np.array], list[float]] = None,
    #     speed_training: typing.Union[None, np.array, float] = None,
    # ) -> np.array:
    #     """get the derivative of the w vector from King paper"""
    #
    #     # by default, get kstar at the training points
    #     if indep_training is None:
    #         indep_training = self.indep_training
    #     if speed_training is None:
    #         speed_training = self.speed_training
    #
    #     # by default, evaluate at the pmf points
    #     direction_data, speed_data = self.get_matrix_direction_speed()
    #
    #     # dkstar in (Neval x Ntr x 2 x Ntr)
    #     dkstar = self._get_dkstar(
    #         direction_data,
    #         speed_data,
    #         direction_ref=indep_training,
    #         speed_ref=indep_training,
    #     )
    #     dw_AEP = dkstar @ (self.get_potential_values().reshape(-1))
    #     return dw_AEP

    # def _get_dKcov(
    #     self,
    #     direction_eval=None,
    #     speed_eval=None,
    # ) -> np.array:
    #     """get the covariance kernel on the training data"""
    #
    #     # by default, get kstar at the training (pmf) points
    #     if direction_eval is None:
    #         direction_eval = self.indep_training
    #     if speed_eval is None:
    #         speed_eval = self.speed_training
    #
    #     dKcov = self._get_dkstar(direction_eval, speed_eval, direction_eval, speed_eval)
    #
    #     return dKcov


class AbstractProductionIntegrator(abc.ABC):
    """
    fidelty-agnostic abstract base class for GP-based AEP integration tools
    """

    @classmethod
    @abc.abstractmethod
    def eval_response_gp(
        self,
        indep_query,
        zscore: float = 0.0,
        return_std=False,
    ) -> np.array:
        """
        return the exploitation response based on the GP at queried points
        """
        pass

    @classmethod
    @abc.abstractmethod
    def eval_production_integrand_gp(
        self,
        zscore: float = 0.0,
        return_std=False,
    ) -> np.array:
        """
        return the production integrand based on the GP at the resource points
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_production_gp(
        self,
        zscore: float = 0.0,
        return_variance: bool = False,
    ) -> float:
        """
        compute a production estimate using the GP integrand

        at a given z-score OR just return mean and the variance
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_d_production_gp(
        self,
    ):
        """
        compute the derivative w.r.t. a production estimate using the GP integrand
        """
        pass


class gpProductionIntegrator(gpProductionToolBase, AbstractProductionIntegrator):
    """
    single-fidelity GP-based production integrator

    parameters
    ----------
    indep_bins : list[np.ndarray]
        list of 1D direction vectors on which the resource is gridded
    potential_bins : np.ndarray
        n-dimensional array of PMF values
    response_function : function [...] -> float (optional)
        function that gives an analytical response functional

    attributes
    ----------
    indep_bins : list[np.ndarray]
        list of 1D independent variables on which the resource is gridded
    potential_bins : np.ndarray
        n-dimensional vector (direction x speed) of PMF values
    response_function : function [...] -> float
        function that gives an analytical response functional (if applicable)
    indep_training : list[np.ndarray]
        storage for indep. var directions (on which GPs are trained, if applicable)
    response_training : np.ndarray
        storage for training responses (on which GPs are trained, if applicable)
    kernel : sklearn.Kernel
        storage for a trained sklearn kernel for the integrand GP (if applicable)
    integrandGP : sklearn.GaussianProcessRegressor
        storage for a trained sklearn Gaussian Process for the integrand GP (if applicable)

    methods
    -------
    set_response_function(pf_in)
        sets the analytical response function (mapping from [indep_vars] -> power)
    get_potential_values()
        get the values of the resource function at bins in the indep vars
    get_indep_mesh()
        get the meshgrid of independent variables
    eval_production_integrand_exact()
        get the integrand of production based on analytical response at the indep var bins
    eval_production_pf()
        get the production value using the analytical response with trapezoidal rule response on the indep bins
    setup_kernel(indep_training, response_training, ...)
        set up a GP kernel for making an estimate of the production
    eval_response_gp(indep_query, zscore=0.0, return_std=False)
        get the response at some set of queried points (optionally with uncertainty)
    eval_production_integrand_gp()
        get the integrand of production based on the GP at the indep bins
    eval_production_gp()
        get the integrated production value using the GP
    """

    def __init__(
        self,
        potential_bins: np.ndarray,
        response_function: typing.Callable[..., float] = None,
    ) -> None:
        super(gpProductionToolBase, self).__init__(potential_bins, response_function)

    def eval_response_gp(
        self,
        indep_query,
        zscore: float = 0.0,
        return_std=False,
    ) -> np.array:
        """
        return the response based on the GP at queried points
        """

        assert not (
            zscore and return_std
        ), "if you are getting the std field, you need to build your zscores yourself"

        # grab the query points
        XYZq = np.vstack([v.flat for v in indep_query]).T

        # evaluate the GP mean and std surfaces
        mean_fcn, std_fcn = self.integrandGP.predict(XYZq, return_std=True)
        mean_fcn = mean_fcn.reshape(indep_query[0].shape)
        std_fcn = std_fcn.reshape(indep_query[0].shape)

        # correct and return as appropriate
        if return_std:
            return [mean_fcn, std_fcn]
        elif not zscore:
            return mean_fcn
        else:
            return mean_fcn + zscore * std_fcn

    def eval_production_integrand_gp(
        self,
        indep_query=None,
        zscore: float = 0.0,
        return_std=False,
    ) -> np.array:
        """
        return the production integrand based on the GP at the resource points
        """

        # default to the resource grid
        if indep_query is None:
            indep_query = self.get_indep_mesh()
            resource_potential_values = self.get_potential_values()
        else:
            raise NotImplementedError("need to figure out how to interpolate the resource grid. -cfrontin")

        # get the gaussian process returns
        power_response_gp = self.eval_response_gp(
            indep_query,
            zscore=zscore,
            return_std=return_std,
        )

        # unpack and return
        if return_std:
            power_response_gp, std_power_response_gp = power_response_gp
            return (
                power_response_gp * resource_potential_values,
                std_power_response_gp * resource_potential_values,
            )
        else:
            return power_response_gp * resource_potential_values

    def get_production_gp(
        self,
        zscore: float = 0.0,
        return_variance: bool = False,
    ) -> float:
        """
        compute an production estimate using the GP integrand

        at a given z-score OR just return mean and the variance
        """

        # get components
        w_AEP = self._get_w()  # at training points
        wTinvK = self._get_wTinvK()  # at training points
        # get Z
        Z_AEP = self._get_Z()

        # compute mean function of AEP
        alpha_AEP = (wTinvK @ self.response_training)
        # and AEP variance
        beta2_AEP = (Z_AEP - (wTinvK @ w_AEP))

        print(
            f"DEBUG!!!!! alpha {alpha_AEP} beta2 {beta2_AEP} "
            + f"beta {np.sqrt(beta2_AEP)} Z {Z_AEP}"
        )

        # handle returns
        if not return_variance:
            return alpha_AEP + zscore * np.sqrt(beta2_AEP)
        else:
            return (alpha_AEP, beta2_AEP)

    def get_d_production_gp(
        self,
        normalized=False,
    ):
        """
        compute the derivative w.r.t. a production estimate using the GP integrand
        """

        raise NotImplementedError("not implemented yet! -cfrontin")

        # # get components
        # wTinvK = self._get_wTinvK()  # at training points
        #
        # norm_factor = (
        #     (1.0 if normalized else self.normalization_power) * 1e6 * 8760 / (1e9)
        # )
        #
        # # compute mean function of AEP
        # dalpha_AEP = (wTinvK) * norm_factor
        #
        # return dalpha_AEP


# class BQDriver(gpProductionToolBase):
#
#     def __init__(
#         self,
#         direction_bins: np.array,
#         speed_bins: np.array,
#         potential_bins: np.array,
#     ) -> None:
#         super(gpProductionToolBase, self).__init__(
#             direction_bins, speed_bins, potential_bins
#         )
#
#     def setup_kernel(
#         self,
#         Prated,
#         Vrated,
#         Ntr=20,
#         kernel=None,
#         train_lengthscales=False,
#         lengthscales=gpProductionToolBase._default_matern_lengthscales,
#         lengthscale_bounds=gpProductionToolBase._default_matern_lengthscalebounds,
#         wrap_kernel=False,
#     ) -> None:
#         # pump fake but representative data to set up the GP fits
#         XYZtr = np.array(self.weibull_sampler(Ntr)).T
#         Ztr = self.default_power_curve(XYZtr, Prated, Vrated)
#
#         return super().setup_kernel(
#             XYZtr[:, 0],
#             XYZtr[:, 1],
#             Ztr,
#             1,
#             Prated,
#             kernel,
#             train_lengthscales,
#             lengthscales,
#             lengthscale_bounds,
#             wrap_kernel,
#         )
#
#     def _get_beta2_obj(
#         self,
#         x_opt: np.array,
#     ) -> float:
#         """objective function for BQ opt. problem"""
#
#         XYZq = x_opt.reshape(-1, 2)
#
#         # the direction/speed meshgrid at pmf points
#         Xd, Yd = self.get_matrix_direction_speed()
#
#         # the direction/speed meshgrid at pmf points
#         Xq, Yq = XYZq[:, 0], XYZq[:, 1]
#
#         # get components
#         w_AEP = self._get_w(indep_training=Xq, speed_training=Yq)
#         wTinvK = self._get_wTinvK(indep_training=Xq, speed_training=Yq)
#
#         norm_factor = 1e6 * self.normalization_power * 8760 / (1e9)
#         # and AEP variance
#         beta2_AEP = -((wTinvK @ w_AEP))  # * norm_factor**2
#
#         return beta2_AEP
#
#     def _get_dbeta2_obj(
#         self,
#         x_opt: np.array,
#         Nturbines: int = 1,
#     ) -> float:
#         """objective function for BQ opt. problem"""
#
#         XYZq = x_opt.reshape(-1, 2)
#
#         # the direction/speed meshgrid at pmf points
#         Xd, Yd = self.get_matrix_direction_speed()
#
#         # the direction/speed meshgrid at pmf points
#         Xq, Yq = XYZq[:, 0], XYZq[:, 1]
#
#         # get components
#         w_AEP = self._get_w(Xq, Yq)
#         wTinvK_AEP = self._get_wTinvK(Xq, Yq)
#         invK = np.linalg.inv(self._get_Kcov(Xq, Yq))
#         d_invK = np.einsum(
#             "ij,mnjk,kl->mnij",
#             invK,
#             self._get_dKcov(Xq, Yq),
#             invK,
#             optimize=True,
#         )
#
#         dbeta2_AEP = 0.0
#         dbeta2_AEP += self._get_dw(Xq, Yq) @ invK @ w_AEP
#         dbeta2_AEP += np.einsum(
#             "i,mnij,j->mn",
#             w_AEP,
#             d_invK,
#             w_AEP,
#             optimize=True,
#         )
#         dbeta2_AEP += np.einsum(
#             "i,mni->mn",
#             wTinvK_AEP,
#             self._get_dw(Xq, Yq),
#             optimize=True,
#         )
#
#         return dbeta2_AEP
#
#     def compute_BQ_points(self, N_BQ, XYZ_init=None, Vcutout=30.0, seed=None):
#         """
#         compute the N_BQ Bayesian quadrature points for the current pdf
#         """
#
#         # default initializer: weibull sampling w/ curtailed mean
#         if XYZ_init is None:
#             XYZ_init = np.vstack(
#                 self.weibull_sampler(
#                     N_BQ,
#                     mean_weibull=Vcutout / 6.0,
#                     seed=seed,
#                 )
#             ).T
#
#         # set up the bounds
#         XYZ_minmax = [
#             (0.0, v)
#             for v in np.vstack(
#                 [
#                     [(2 * np.pi) for v in range(N_BQ)],
#                     [(Vcutout) for v in range(N_BQ)],
#                 ]
#             )
#             .T.flatten()
#             .tolist()
#         ]
#
#         obj_fun = lambda x: self._get_beta2_obj(x)
#
#         res_opt = opt.minimize(
#             obj_fun,
#             XYZ_init.flatten(),
#             bounds=XYZ_minmax,
#             # method="nelder-mead",
#             # method="powell",
#             # method="cobyla",
#             # method="slsqp",
#             method="l-bfgs-b",
#             # method="newton-krylov",
#             options={
#                 "maxiter": 1000,
#             },
#             # jac= lambda x: integrator._get_dbeta2_obj(x),
#         )
#         XYZ_opt = res_opt.x.reshape(-1, 2)
#
#         return XYZ_opt
