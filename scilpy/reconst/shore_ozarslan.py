import warnings
import numpy as np
from dipy.reconst.cache import Cache
from dipy.core.geometry import cart2sphere
from warnings import warn
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import genlaguerre, gamma
from scipy.misc import factorial
from dipy.reconst.shm import real_sph_harm
from cvxopt import matrix, solvers
import dipy.reconst.dti as dti
from dipy.utils.optpkg import optional_package
cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class ShoreOzarslanModel(Cache):
    r""" Analytical and continuous modeling of the diffusion signal using
        the SHORE basis [1].
        This implementation is based on Appendix [1] with included
        Laplacian regularization [2].

        The main idea is to model the diffusion signal as a linear
        combination of continuous functions $\phi_i$,

        ..math::
            :nowrap:
                \begin{equation}
                    S(\mathbf{q})= \sum_{i=0}^I  c_{i} \phi_{i}(\mathbf{q}).
                \end{equation}

        where $\mathbf{q}$ is the wavector which corresponds to different
        gradient directions.

        From the $c_i$ coefficients, there exists an analytical formula to
        estimate the ODF, RTOP, RTAP, RTPP and MSD. It is also possible to
        compute the isotropic and anisotropic propagator.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class. The bvalues
            should be in the normal s/mm^2. If you have them, please give
            big_delta and small_delta in seconds.
        radial_order : unsigned int,
            an even integer that represent the order of the basis
        laplacian_regularization: bool,
            Regularize using the Laplacian of the SHORE basis.
        laplacian_weighting: string or scalar,
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight [3]. A scalar sets the regularization
            weight to that value.
        separated_regularization: bool,
            Regularize using angular Laplace Beltrami
            and radial low-pass filter.
        separated_weighting : string or scalar,
            If 'GCV' it uses generalized cross-validation [3]. If a scalar is
            used it sets this weight for both the radial and angular
            regularization.
        tau : float,
            diffusion time. Defined as $\Delta-\delta/3$ in seconds.
            Default value makes q equal to the square root of the b-value.
        constrain_e0 : bool,
            Constrain the optimization such that E(0) = 1.
        positive_constraint : bool,
            Constrain the propagator to be positive.
        pos_grid : int,
            Grid that define the points of the EAP in which we want to enforce
            positivity.
        pos_radius : float,
            Radius of the grid of the EAP in which enforce positivity in
            millimeters. By default 20e-03 mm.
        tensor_scale_lower_bound : float,
            Sets lower limit on the calculated SHORE scale parameters. Its main
            purpose is to remove negative eigenvalues.
        tensor_linearity_threshold : float,
            Sets lower limit on linearity of the tensor estimated on the data.
            This threshold exists to ensure that the automated RTAP and RTPP
            computation, where the main tensor eigenvector is used as a
            direction, does not use this direction when the tensor is not
            cigar shaped.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.
           [2] Fick, In Preparation.
           [3] Craven et al. "Smoothing Noisy Data with Spline Functions."
           NUMER MATH 31.4 (1978): 377-403.
        """

    def __init__(self,
                 gtab,
                 radial_order=6,
                 laplacian_regularization=False,
                 laplacian_weighting='GCV',
                 separated_regularization=False,
                 separated_weighting='GCV',
                 lambdaN=None,
                 lambdaL=None,
                 constraint_e0=False,
                 positive_constraint=False,
                 pos_grid=10,
                 pos_radius=20e-03,
                 tensor_scale_lower_bound=1e-04,
                 tensor_linearity_threshold=0.3):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        if radial_order >= 0 and radial_order % 2 == 0:
            self.radial_order = radial_order
        else:
            msg = "radial_order must be a non-zero even positive number."
            raise ValueError(msg)
        self.laplacian_regularization = laplacian_regularization
        self.constraint_e0 = constraint_e0
        self.positive_constraint = positive_constraint
        self.separated_regularization = separated_regularization
        self.pos_grid = pos_grid
        self.tensor_scale_lower_bound = tensor_scale_lower_bound
        self.tensor_linearity_threshold = tensor_linearity_threshold

        if laplacian_regularization and separated_regularization:
            msg = "Cannot use both Laplacian and Separated regularization.\
                   Please choose one."
            raise ValueError(msg)

        if laplacian_weighting == 'GCV' or \
                np.isscalar(laplacian_weighting):
            if np.isscalar(laplacian_weighting) and \
                    laplacian_weighting < 0.:
                msg = "Laplacian Regularization weighting must be positive."
                raise ValueError(msg)
            else:
                self.laplacian_weighting = laplacian_weighting
            self.laplacian_matrix = shore_laplace_reg_matrix(radial_order, 1)

        if separated_weighting == 'GCV' or \
                np.isscalar(separated_weighting):
            if np.isscalar(separated_weighting) and \
                    separated_weighting < 0:
                msg = "Separated Regularization weighting must be positive."
            else:
                self.separated_weighting = separated_weighting
            self.N_mat, self.L_mat = shore_separated_reg_matrices(radial_order)

        if self.positive_constraint:
            self.constraint_e0 = True
            if not have_cvxopt:
                raise ValueError(
                    'CVXOPT package needed to enforce constraints')

            constraint_grid = gen_rgrid(pos_radius, pos_grid)
            self.constraint_grid = constraint_grid
            self.pos_K_independent = shore_K_mu_independent(radial_order,
                                                            constraint_grid)

        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0

        ind_mat = shore_index_matrix(radial_order)
        self.ind_mat = ind_mat

        qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
        q = gtab.bvecs * qvals[:, None]
        self.Q_mu_independent = shore_Q_mu_independent(self.radial_order, q)

        self.number_of_coefficients = number_of_coef(radial_order) 
        self.B_mat = B_matrix(radial_order)
        self.rtpp_vec = rtpp_vector(radial_order, ind_mat)
        self.rtap_vec = rtap_vector(radial_order, ind_mat)
        self.rtop_vec = rtop_vector(radial_order, ind_mat, self.B_mat)
        self.msd_vec = msd_vector(radial_order, ind_mat, self.B_mat)
        self.tenmodel = dti.TensorModel(gtab)

    @multi_voxel_fit
    def fit(self, data, mask=None):

        tenfit = self.tenmodel.fit(data, mask)
        evals = tenfit.evals
        tensor_linearity = dti.linearity(evals)
        if tensor_linearity > self.tensor_linearity_threshold:
            R = tenfit.evecs
        else:
            R = None

        # improves robustness of the fitting to include a
        # lower bound on eigenvalues.
        evals = np.clip(evals, self.tensor_scale_lower_bound, evals.max())
        mu = np.sqrt(evals.mean() * 2 * self.tau)
        qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)

        Q_mu_dependent = shore_Q_mu_dependent(self.radial_order, mu, qvals)

        M = Q_mu_dependent * self.Q_mu_independent

        if not self.positive_constraint and not self.constraint_e0 and not\
            self.laplacian_regularization and not\
                self.separated_regularization:
                coef = np.dot(np.dot(np.linalg.pinv((np.dot(M.T,
                                                            M))), M.T), data)

        if self.laplacian_regularization:
            laplacian_matrix = self.laplacian_matrix * mu

            if self.laplacian_weighting == 'GCV':
                lopt = generalized_crossvalidation(data, M,
                                                   laplacian_matrix)
            else:
                lopt = self.laplacian_weighting

            if self.constraint_e0 or self.positive_constraint:
                Q = matrix(np.dot(M.T, M) + lopt * laplacian_matrix)
            else:
                Mreg = np.dot(M.T, M) + lopt * laplacian_matrix
                zeros = np.zeros(Mreg.shape[0])
                coef = zeros

                if (data == 0).all() :
                    coef = zeros
                else :
                    try:
                        MregInv = np.linalg.pinv(Mreg)
                        coef = np.dot(np.dot(MregInv, M.T), data)
                    except np.linalg.linalg.LinAlgError as err: 
                        if 'SVD did not converge' in err.message:
                            warnings.warn('SVD did not converge')
                            coef = zeros
                        else:
                            raise

        else:
            lopt = 0

        if self.separated_regularization:
            N_mat = self.N_mat
            L_mat = self.L_mat
            if self.separated_weighting == 'GCV':
                loptN, loptL = generalized_crossvalidation2D(
                    data, M, N_mat, L_mat)
            else:
                loptN = loptL = self.separated_weighting

            if self.constraint_e0 or self.positive_constraint:
                Q = matrix(np.dot(M.T, M) + loptN * N_mat + loptL * L_mat)
            else:
                Mreg = np.dot(M.T, M) + loptN * N_mat + loptL * L_mat
                zeros = np.zeros(Mreg.shape[0])
                coef = zeros

                if (data == 0).all() :
                    coef = zeros
                else :
                    try:
                        MregInv = np.linalg.pinv(Mreg)
                        coef = np.dot(np.dot(MregInv, M.T), data)
                    except np.linalg.linalg.LinAlgError as err: 
                        if 'SVD did not converge' in err.message:
                            warnings.warn('SVD did not converge')
                            coef = zeros
                        else:
                            raise

        else:
            loptN = loptL = 0

        if self.constraint_e0 or self.positive_constraint:
            Q = matrix(np.dot(M.T, M))

            if self.positive_constraint:
                constraint_grid = self.constraint_grid
                K_independent = self.pos_K_independent
                K_dependent = shore_K_mu_dependent(self.radial_order,
                                                   mu, constraint_grid)
                K = K_independent * K_dependent
                G = matrix(-1*K)
                h = matrix(np.zeros((K.shape[0])), (K.shape[0], 1))
            else:
                G = None
                h = None
            if self.constraint_e0:
                data = data / data[self.gtab.b0s_mask].mean()
                p = matrix(-1 * np.dot(M.T, data))
                A = matrix(M[0], (1, M.shape[1]))
                b = matrix(1.0)
            else:
                p = matrix(-1 * np.dot(M.T, data))
                A = None
                b = None
            solvers.options['show_progress'] = False

            try:
                sol = solvers.qp(Q, p, G, h, A, b)
                coef = np.array(sol['x'])[:, 0]
            except ValueError:
                coef = np.zeros(self.ind_mat.shape[0])

        if not(self.constraint_e0):
            coef = coef / sum(coef * self.B_mat)
        
        # Apply the mask to the coefficients
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            coef *= mask[..., None]

        return ShoreOzarslanFit(self, coef, mu, R, lopt, loptN, loptL)


class ShoreOzarslanFit():

    def __init__(self, model, shore_coef, mu, R, lopt, loptN, loptL):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
        AnalyticalModel
        shore_coef : 1d ndarray,
        shore coefficients
        mu : float,
        scaling factor
        R : 3x3 numpy array,
        tensor eigenvectors
        lopt : float,
        laplacian regularization weight
        loptN : float,
        separated regularization radial weight
        loptL : float,
        separated regularization angular weight
        """

        self.model = model
        self._shore_coef = shore_coef
        self.gtab = model.gtab
        self.radial_order = model.radial_order
        self.mu = mu
        self.R = R
        self.lopt = lopt
        self.loptN = loptN
        self.loptL = loptL

    @property
    def shore_coeff(self):
        """The SHORE coefficients
        """
        return self._shore_coef

    def odf(self, sphere, radial_moment=0):
        r""" Calculates the real analytical odf for a given discrete sphere.
        Computes the design matrix of the ODF for the given sphere vertices
        and radial moment. The radial moment s acts as a
        sharpening method [1,2].
        ..math::
            :nowrap:
                \begin{equation}\label{eq:ODF}
                    \textrm{ODF}_s(\textbf{u}_r)=\int r^{2+s}
                    P(r\cdot \textbf{u}_r)dr.
                \end{equation}
        """
        I = self.model.cache_get('ODF_matrix', key=(sphere, radial_moment))

        if I is None:
            I = shore_odf_matrix(self.radial_order, 1,
                                 radial_moment, sphere.vertices)
            self.model.cache_set('ODF_matrix', (sphere, radial_moment), I)

        odf = self.mu ** (radial_moment) * np.dot(I, self._shore_coef)
        #return np.clip(odf, 0, odf.max())
        return odf

    def fitted_signal(self, gtab=None):
        """ Recovers the fitted signal. If no gtab is given it recovers
        the signal for the gtab of the data.
        """
        if gtab is None:
            E = self.signal(self.model.gtab)
        else:
            E = self.signal(gtab)
        return E

    def signal(self, qvals_or_gtab):
        r'''Recovers the reconstructed signal for any qvalue array or
        gradient table. We precompute the mu independent part of the
        design matrix Q to speed up the computation.
        '''
        if isinstance(qvals_or_gtab, np.ndarray):
            q = qvals_or_gtab
            qvals = np.sqrt((q ** 2).sum(-1))
            if not q.flags.writeable:
                key = q.data
            else:
                key = None
        else:
            gtab = qvals_or_gtab
            qvals = np.sqrt(gtab.bvals / self.model.tau) / (2 * np.pi)
            q = qvals[:, None] * gtab.bvecs
            key = gtab

        if key is not None:
            Q_independent = self.model.cache_get('Q_sample_matrix', key=key)
        else:
            Q_independent = None

        if Q_independent is None:
            Q_independent = shore_Q_mu_independent(self.radial_order, q)
            if key is not None:
                self.model.cache_set('Q_sample_matrix', key, Q_independent)

        Q_dependent = shore_Q_mu_dependent(self.radial_order, self.mu, qvals)

        Q = Q_independent * Q_dependent

        E = np.dot(Q, self._shore_coef)
        return E

    def pdf(self, r_points):
        """ Diffusion propagator on a given set of real points.
        if the array r_points is non writeable, then intermediate
        results are cached for faster recalculation. r must be given in mm.
        """
        if not r_points.flags.writeable:
            key = r_points.data
            K_independent = self.model.cache_get(
                'K_matrix_independent', key=key)

            if K_independent is None:
                K_independent = shore_K_mu_independent(self.radial_order,
                                                       r_points)
                self.model.cache_set('shore_K_independent', key, K_independent)
                K_dependent = shore_K_mu_dependent(self.radial_order,
                                                   self.mu, r_points)
                K = K_independent * K_dependent

        else:
            K = shore_psi_matrix(self.radial_order, self.mu, r_points)

        P = np.dot(K, self._shore_coef)
        return P

    def rtpp(self, direction=None):
        '''
        Recovers the directional Return To Plane Probability (RTPP) [1,2].
        Its value is only valid along the direction of the fiber. If no
        direction is given the biggest eigenvector of the tensor is used.
        Its value is defined as:
        ..math::
          :nowrap:
              \begin{equation}
                RTPP=\int_{\mathbb{R}^2}P(\textbf{r}_{\perp})
                d\textbf{r}_{\perp}=\frac{1}{(2\pi)^{1/2}u_0}
                \sum_{N=0}^{N_{max}}\sum_{\{j,l,m\}}\textbf{c}_{\{j,l,m\}}
                (-1)^{j-1}2^{-l/2}\kappa(j,l)Y_l^m(\textbf{u}_{fiber})
              \end{equation}
        The vector is precomputed to speed up the computation.
        '''
        if direction is None:
            if self.R is None:
                warn('Tensor linearity too low to use main tensor eigenvalue '
                     'as fiber direction. Returning 0.')
                return 0.
            else:
                direction = np.array(self.R[:, 0], ndmin=2)
                r, theta, phi = cart2sphere(direction[:, 0], direction[:, 1],
                                            direction[:, 2])

        else:
            r, theta, phi = cart2sphere(direction[0], direction[1],
                                        direction[2])

        inx = self.model.ind_mat
        rtpp_vec = self.model.rtpp_vec

        rtpp = self._shore_coef * (1 / self.mu) *\
            rtpp_vec * real_sph_harm(inx[:, 2], inx[:, 1], theta, phi)

        return rtpp.sum()

    def rtap(self, direction=None):
        """ Recovers the directional Return To Axis Probability (RTAP) [1,2].
        Its value is only valid along the direction of the fiber. If no
        direction is given the biggest eigenvector of the tensor is used.
        Its value is defined as:
        ..math::
          :nowrap:
              \begin{equation}
              RTAP=\int_{\mathbb{R}}P(\textbf{r}_{\parallel})
              d\textbf{r}_{\parallel}=\frac{1}{(2\pi) u_0^2}
              \sum_{N=0}^{N_{max}}\sum_{\{j,l,m\}}\textbf{c}_{\{j,l,m\}}
              (-1)^{j-1}2^{-l/2}\kappa(j,l)Y_l^m(\textbf{u}_{fiber})
        The vector is precomputed to speed up the computation.
        """
        if direction is None:
            if self.R is None:
                warn('Tensor linearity too low to use main tensor eigenvalue '
                     'as fiber direction. Returning 0.')
                return 0.
            else:
                direction = np.array(self.R[:, 0], ndmin=2)
                r, theta, phi = cart2sphere(direction[:, 0],
                                            direction[:, 1], direction[:, 2])
        else:
            r, theta, phi = cart2sphere(direction[0], direction[1],
                                        direction[2])

        inx = self.model.ind_mat
        rtap_vec = self.model.rtap_vec
        rtap = self._shore_coef * (1 / self.mu ** 2) *\
            rtap_vec * real_sph_harm(inx[:, 2], inx[:, 1], theta, phi)
        return rtap.sum()

    def rtop(self):
        r""" Recovers the Return To Origin Probability (RTOP) [1,2].
        Its value is a general property of the fitted signal, defined as:
        ..math::
            :nowrap:
                \begin{equation}
                    RTOP=P(0)=\frac{1}{(2\pi)^{3/2}u_0^3}
                    \sum_{N=0}^{N_{max}}\sum_{\{j,l,m\}}\,
                    \textbf{c}_{\{j,l,m\}}(-1)^{j-1}L_{j-1}^{1/2}(0)
                    \delta_{(l,0)}
                \end{equation}
        The vector is precomputed to speed up the computation.
        """
        rtop_vec = self.model.rtop_vec
        rtop = (1 / self.mu ** 3) * rtop_vec * self._shore_coef
        return rtop.sum()

    def msd(self):
        """Recovers the Mean Squared Displacement (MSD) [2].
        Its value is a general propoerty of the fitted signal, defined as:
            ..math::
                :nowrap:
                    \begin{equation}
                        MSD=u_0^2\sum_{N=0}^{N_{max}}\sum_{\{j,l,m\}}
                        \textbf{c}_{\{j,l,m\}}(4j-1)L_{j-1}^{1/2}(0)
                        \delta_{(l,0)}.
                    \end{equation}
        The vector is precomputed to speed up the computation.
        """
        msd_vec = self.model.msd_vec
        msd = self.mu ** 2 * msd_vec * self._shore_coef
        return msd.sum()

    def anisotropic_propagator(self):
        """Returns a propagator with only the anisotropic components
        """
        ind_mat = shore_index_matrix(self.radial_order)
        isotropic_coef = ind_mat[:, 1] == 0
        coeff = self._shore_coef.copy()
        coeff[isotropic_coef] = 0
        return ShoreOzarslanFit(self.model, coeff, self.mu,
                                self.R, self.lopt, self.loptN, self.loptL)

    def isotropic_propagator(self):
        """Returns a propagator with only the isotropic components
        """
        ind_mat = shore_index_matrix(self.radial_order)
        anisotropic_coef = ind_mat[:, 1] != 0
        coeff = self._shore_coef.copy()
        coeff[anisotropic_coef] = 0
        return ShoreOzarslanFit(self.model, coeff, self.mu,
                                self.R, self.lopt, self.loptN, self.loptL)

    def propagator_anisotropy(self, epsilon=0.4):
        r"""Implements the propagator anisotropy from [1].
        """
        iso = self.isotropic_propagator()._shore_coef
        cos_theta = np.nan_to_num(np.dot(self._shore_coef, iso) / np.sqrt(
            np.dot(self._shore_coef, self.shore_coeff) * np.dot(iso, iso)))

        sin_theta = np.sqrt(1 - cos_theta ** 2)
        return sin_theta ** (3 * epsilon) /\
            (1. - 3 * sin_theta ** epsilon + 3 * sin_theta ** (2 * epsilon))


def shore_index_matrix(radial_order):
    """Computes the SHORE basis order indices according to [1].
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            for m in range(-1 * (n + 2 - 2 * j), (n + 3 - 2 * j)):
                index_matrix.append([j, n + 2 - 2 * j, m])

    return np.array(index_matrix)


def rtpp_vector(radial_order, ind_mat):
    '''Precomputes the vector that is used for th RTPP computation.
    '''
    rtpp_vec = np.zeros((ind_mat.shape[0]))
    count = 0
    for n in range(0, radial_order + 1, 2):
            for j in range(1, 2 + n / 2):
                l = n + 2 - 2 * j
                const = (-1/2.0) ** (l/2) / np.sqrt(np.pi)
                matsum = 0
                for k in range(0, j):
                    matsum += (-1) ** k * \
                        binomialfloat(j + l - 0.5, j - k - 1) *\
                        gamma(l / 2 + k + 1 / 2.0) /\
                        (factorial(k) * 0.5 ** (l / 2 + 1 / 2.0 + k))
                for m in range(-l, l + 1):
                    rtpp_vec[count] = const * matsum
                    count += 1
    return rtpp_vec


def rtap_vector(radial_order, ind_mat):
    '''Precomputes the vector that is used for th RTAP computation.
    '''
    rtap_vec = np.zeros((ind_mat.shape[0]))
    s = -2
    count = 0

    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0)) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum = matsum + ((-1) ** k *
                                   binomialfloat(j + l - 0.5, j - k - 1) *
                                   gamma((l + s + 3) / 2.0 + k)) /\
                    (factorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                rtap_vec[count] = kappa * matsum
                count += 1

    return 2 * rtap_vec


def kappa(j, l):
    """Helper function of RTPP/RTAP computation.
    """
    kappasum = 0
    for k in range(0, j):
        kappasum += (-1) ** k / factorial(k) *\
            binomialfloat(j + l - 0.5, j - k - 1) * \
            gamma(((l + 1) / 2.0 + k)) /\
            2 ** (l / 2.0 + k)


def rtop_vector(radial_order, ind_mat, B_mat):
    '''Precomputes the vector that is used for th RTOP computation.
    '''
    const = 1 / (2 * np.sqrt(2.0) * np.pi ** (3 / 2.0))
    rtop_vec = const * (-1.0) ** (ind_mat[:, 0]-1) * B_mat
    return rtop_vec


def msd_vector(radial_order, ind_mat, B_mat):
    '''Precomputes the vector that is used for th MSD computation.
    '''
    msd_vec = -(1 - 4 * ind_mat[:, 0]) * B_mat
    return msd_vec


def shore_Q_mu_independent(radial_order, q):
    r'''Computed the u0 independent part of the design matrix.
    ..math::
        :nowrap:
            \begin{align}
                \Xi_{jlm(i)}(u_0,\textbf{q}_k)=&\overbrace{u_0^{l(i)}
                e^{-2\pi^2u_0^2q_k^2}L_{j(i)-1}^{l(i)+1/2}
                (4\pi^2u_0^2q_k^2)}^{u_0\,dependent}
                \overbrace{\sqrt{4\pi}i^{-l(i)}(2\pi^2q_k^2)^{l(i)/2}
                Y_{l(i)}^{m(i)}(\textbf{u}_{q_k})}^{u_0\,independent}
                =&A_{jl(i)}(q_k)B_{lm(i)}(\textbf{q}_k)
            \end{align}
    '''
    ind_mat = shore_index_matrix(radial_order)

    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1],
                                   q[:, 2])
    theta[np.isnan(theta)] = 0

    n_elem = ind_mat.shape[0]
    n_rgrad = theta.shape[0]
    Q_mu_independent = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = np.sqrt(4 * np.pi) * (-1) ** (-l / 2) * \
                (2 * np.pi ** 2 * qval ** 2) ** (l / 2)
            for m in range(-1 * (n + 2 - 2 * j), (n + 3 - 2 * j)):
                Q_mu_independent[:, counter] = const * \
                    real_sph_harm(m, l, theta, phi)
                counter += 1
    return Q_mu_independent


def shore_Q_mu_dependent(radial_order, mu, qval):
    '''Computed the u0 dependent part of the design matrix.
    See shore_Q_mu_independent for help.
    '''
    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = qval.shape[0]
    Q_u0_dependent = np.zeros((n_qgrad, n_elem))

    counter = 0

    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = mu ** l * np.exp(-2 * np.pi ** 2 * mu ** 2 * qval ** 2) *\
                genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * mu ** 2 *
                                            qval ** 2)
            for m in range(-l, l + 1):
                Q_u0_dependent[:, counter] = const
                counter += 1

    return Q_u0_dependent


def shore_phi_matrix(radial_order, mu, q):
    '''Computed the Q matrix completely without separation into
    mu-depenent / -independent. See shore_Q_mu_independent for help.
    '''
    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1],
                                   q[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = q.shape[0]
    M = np.zeros((n_qgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (l / 2) * np.sqrt(4.0 * np.pi) *\
                (2 * np.pi ** 2 * mu ** 2 * qval ** 2) ** (l / 2) *\
                np.exp(-2 * np.pi ** 2 * mu ** 2 * qval ** 2) *\
                genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * mu ** 2 *
                                            qval ** 2)
            for m in range(-l, l+1):
                M[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1
    return M


def shore_K_mu_independent(radial_order, rgrad):
    '''Computes mu independent part of K [2]. Same trick as with Q.
    '''
    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (j - 1) *\
                (np.sqrt(2) * np.pi) ** (-1) *\
                (r ** 2 / 2) ** (l / 2)
            for m in range(-l, l+1):
                K[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1
    return K


def shore_K_mu_dependent(radial_order, mu, rgrad):
    '''Computes mu dependent part of K [2]. Same trick as with Q.
    '''
    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (mu ** 3) ** (-1) * mu ** (-l) *\
                np.exp(-r ** 2 / (2 * mu ** 2)) *\
                genlaguerre(j - 1, l + 0.5)(r ** 2 / mu ** 2)
            for m in range(-l, l + 1):
                K[:, counter] = const
                counter += 1
    return K


def shore_psi_matrix(radial_order, mu, rgrad):
    """Computes the K matrix without optimization.
    """

    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (j - 1) * \
                    (np.sqrt(2) * np.pi * mu ** 3) ** (-1) *\
                    (r ** 2 / (2 * mu ** 2)) ** (l / 2) *\
                np.exp(- r ** 2 / (2 * mu ** 2)) *\
                genlaguerre(j - 1, l + 0.5)(r ** 2 / mu ** 2)
            for m in range(-l, l + 1):
                K[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1

    return K


def binomialfloat(n, k):
    """Custom Binomial function
    """
    return factorial(n) / (factorial(n - k) * factorial(k))


def shore_odf_matrix(radial_order, mu, s, vertices):
    r"""The ODF in terms of SHORE coefficients for arbitrary radial moment
    can be given as [2]:
    ..math::
        :nowrap:
            \begin{equation}
                ODF_s(u_0,\textbf{v})=\sum_{N=0}^{N_{max}}
                \sum_{\{j,l,m\}}\textbf{c}_{\{j,l,m\}}
                \Omega_s^{jlm}(u_0,\textbf{v})
            \end{equation}

    with $\textbf{v}$ an orientation on the unit sphere and the ODF
    basis function:
    ..math::
        :nowrap:
            \begin{equation}
                \Omega_s^{jlm}(u_0,\textbf{v})=\frac{u_0^s}{\pi}(-1)^{j-1}
                2^{-l/2}\kappa(j,l,s)Y^l_m(\textbf{v})
            \end{equation}
    with
    ..math::
        :nowrap:
            \begin{equation}
                \kappa(j,l,s)=\sum_{k=0}^{j-1}\frac{(-1)^k}{k!}
                \binom{j+l-1/2}{j-k-1}
                \frac{\Gamma((l+s+3)/2+k)}{2^{-((l+s)/2+k)}}.
            \end{equation}
    """
    r, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1],
                                vertices[:, 2])

    theta[np.isnan(theta)] = 0
    ind_mat = shore_index_matrix(radial_order)
    n_vert = vertices.shape[0]
    n_elem = ind_mat.shape[0]
    odf_mat = np.zeros((n_vert, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0) * mu ** s) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum += ((-1) ** k * binomialfloat(j + l - 0.5, j - k - 1) *
                           gamma((l + s + 3) / 2.0 + k)) /\
                    (factorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                odf_mat[:, counter] = kappa * matsum *\
                    real_sph_harm(m, l, theta, phi)
                counter += 1

    return odf_mat


def shore_laplace_reg_matrix(radial_order, mu):
    r'''
    The Laplacian regularization matrix [2] is given as.
    ..math::
        :nowrap:
            \begin{equation}
                \textbf{R}_{ik}(u_0)=\delta_{(l(i),l(k))}
                \delta_{(m(i),m(k))}R\left(u_0,j(i),j(k),l\right)
            \end{equation}

    where we define the intermediate function $R$ as
    ..math::
        :nowrap:
            \begin{equation}
                R\left(u_0,j(i),j(k),l\right)=u_0
                \begin{cases}
                    \delta_{(j(i),j(k)+2)}\frac{2^{2-l}\pi^2
                    \Gamma(\frac{5}{2}+j(k)+l)}{\Gamma(j(k))} & \\
                    \delta_{(j(i),j(k)+1)}\frac{2^{2-l}\pi^2 (-3+4j(i)+2l)
                    \Gamma(\frac{3}{2}+j(k)+l)}{\Gamma(j(k))} & \\
                    \delta_{(j(i),j(k))}\frac{2^{-l}\pi^2
                    \left(3+24j(i)^2+4(-2+l)l+12j(i)(-1+2l)\right)
                    \Gamma(\frac{1}{2}+j(i)+l)}{\Gamma(j(i))} & \\
                    \delta_{(j(i),j(k)-1)}\frac{2^{2-l}\pi^2 (-3+4j(k)+2l)
                    \Gamma(\frac{3}{2}+j(i)+l)}{\Gamma(j(i))} & \\
                    \delta_{(j(i),j(k)-2)}\frac{2^{2-l}\pi^2
                    \Gamma(\frac{5}{2}+j(i)+l)}{\Gamma(j(i))} &
                \end{cases}
            \end{equation}
    '''
    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                jk = ind_mat[k, 0]
                l = ind_mat[i, 1]
                if ji == (jk + 2):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + jk + l) / gamma(jk)
                elif ji == (jk + 1):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * ji + 2 * l) * gamma(3 / 2.0 + jk + l) /\
                        gamma(jk)
                elif ji == jk:
                    LR[i, k] = 2 ** (-l) * np.pi ** 2 * mu *\
                        (3 + 24 * ji ** 2 + 4 * (-2 + l) *
                         l + 12 * ji * (-1 + 2 * l)) *\
                        gamma(1 / 2.0 + ji + l) / gamma(ji)
                elif ji == (jk - 1):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * jk + 2 * l) * gamma(3 / 2.0 + ji + l) /\
                        gamma(ji)
                elif ji == (jk - 2):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + ji + l) / gamma(ji)

    return LR


def shore_separated_reg_matrices(radial_order):
    ind_mat = shore_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    N_mat = np.zeros((n_elem, n_elem))
    L_mat = np.zeros((n_elem, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            for m in range(-l, l+1):
                N_mat[counter, counter] = (n * (n + 1)) ** 2
                L_mat[counter, counter] = (l * (l + 1)) ** 2
                counter += 1

    return N_mat, L_mat


def B_matrix(radial_order):
    """ Helper function for RTOP.
    """
    ind_mat = shore_index_matrix(radial_order)

    b_mat = np.zeros((ind_mat.shape[0]))
    for i in range(ind_mat.shape[0]):
        if ind_mat[i, 1] == 0:
            b_mat[i] = genlaguerre(ind_mat[i, 0] - 1, 0.5)(0)

    return b_mat


def gen_rgrid(rmax, Nstep=10):
    rgrad = []
    # Build a regular grid of Nstep**3 points in (R^2 X R+)
    for xx in np.linspace(-rmax, rmax, 2 * Nstep + 1):
        for yy in np.linspace(-rmax, rmax, 2 * Nstep + 1):
            for zz in np.linspace(0, rmax, Nstep+1):
                rgrad.append([xx, yy, zz])
    return np.array(rgrad)

def number_of_coef(radial_order):
    F = radial_order / 2
    return (F + 1) * (F + 2) * (4 * F + 3) / 6


def generalized_crossvalidation(data, M, LR):
    """Generalized Cross Validation Function [3]
    """
    #lrange = np.hstack([10 * np.linspace(0, 0.1, 20)[1:] ** 2,
                        #np.linspace(0.2, 1, 9),
                        #np.linspace(2, 10, 9)])[1:]
    lrange = np.linspace(0,1,40)[1:]
    samples = lrange.shape[0]
    MMt = np.dot(M.T, M)
    K = len(data)
    gcvold = gcvnew = 10e10
    i = -1
    while gcvold >= gcvnew and i < samples - 2:
        gcvold = gcvnew
        i = i + 1
        S = np.dot(np.dot(M, np.linalg.pinv(MMt + lrange[i] * LR)), M.T)
        trS = np.matrix.trace(S)
        normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
        gcvnew = normyytilde / (K - trS)

    return lrange[i-1]


def generalized_crossvalidation2D(data, M, N_mat, L_mat):
    """Generalized Cross Validation in 2D [3]
    """
    lrangeN = (10 ** (-2) * np.linspace(0, 10, 16) ** 2)[1:]
    lrangeL = (10 ** (-2) * np.linspace(0, 10, 16) ** 2)[1:]
    gcv = np.zeros(([15, 15]))
    MMt = np.dot(M.T, M)
    K = len(data)
    for i in range(len(lrangeN)):
        for j in range(len(lrangeN)):
            S = np.dot(np.dot(M, np.linalg.pinv(MMt + lrangeN[i] * N_mat
                                                + lrangeL[j] * L_mat)), M.T)
            trS = np.matrix.trace(S)
            normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
            gcv[i, j] = normyytilde / (K - trS)
    gcvmin = np.unravel_index(np.argmin(gcv), (15, 15))
    return lrangeN[gcvmin[0]], lrangeL[gcvmin[1]]
