#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module contains the core classes and functions to perform the
Brownian motion and timestamps simulation.
"""

import os
import hashlib
import itertools
from pathlib import Path
from time import ctime
import json

import numpy as np
from numpy import array, sqrt
import tables

from .storage import TrajectoryStore, TimestampStore, ExistingArrayError
from .iter_chunks import iter_chunksize, iter_chunk_index
from .psflib import NumericPSF, GaussianPSF, psf_from_pytables

from ._version import get_versions
__version__ = get_versions()['version']

# Avogadro constant
NA = 6.022141e23    # [mol^-1]


def get_seed(seed, ID=0, EID=0):
    """Get a random seed that is a combination of `seed`, `ID` and `EID`.
    Provides different, but deterministic, seeds in parallel computations
    """
    return seed + EID + 100 * ID


def hashfunc(x):
    return hashlib.sha1(repr(x).encode()).hexdigest()


class Box:
    """The simulation box. Sizes in meters."""
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2
        self.b = array([[x1, x2], [y1, y2], [z1, z2]])

    def to_dict(self):
        return {'x1': self.x1, 'x2': self.x2,
                'y1': self.y1, 'y2': self.y2,
                'z1': self.z1, 'z2': self.z2}

    def to_json(self):
        return json.dumps(self.to_dict())

    @property
    def volume(self):
        """Box volume in m^3."""
        return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

    @property
    def volume_L(self):
        """Box volume in liters."""
        return self.volume * 1e3

    def __repr__(self):
        return u"Box: X %.1fum, Y %.1fum, Z %.1fum" % (
            (self.x2 - self.x1) * 1e6,
            (self.y2 - self.y1) * 1e6,
            (self.z2 - self.z1) * 1e6)


class Particle(object):
    """Class to describe a single particle."""
    def __init__(self, D, x0, y0, z0):
        self.D = D   # diffusion coefficient in SI units, m^2/s
        self.x0, self.y0, self.z0 = x0, y0, z0

    @property
    def r0(self):
        return np.array([self.x0, self.y0, self.z0])

    def __eq__(self, other):
        return (self.r0 == other.r0).all() and self.D == other.D

    def to_dict(self):
        return {'D': self.D, 'x0': self.x0, 'y0': self.y0, 'z0': self.z0}


class Particles(object):
    """A list of Particle() objects and a few attributes."""

    @staticmethod
    def _generate(num_particles, D, box, rs):
        """Generate a list of `Particle` objects."""
        X0 = rs.rand(num_particles) * (box.x2 - box.x1) + box.x1
        Y0 = rs.rand(num_particles) * (box.y2 - box.y1) + box.y1
        Z0 = rs.rand(num_particles) * (box.z2 - box.z1) + box.z1
        return [Particle(D=D, x0=x0, y0=y0, z0=z0)
                for x0, y0, z0 in zip(X0, Y0, Z0)]

    @staticmethod
    def from_specs(num_particles, D, box, rs=None, seed=1):
        """
        Create particles populations in a single-step.

        Arguments:
            num_particles (sequence): contains the number of particles
                in each population
            D (sequence): contains the diffusion. coefficient (m^2 / s)
                for each population.
            box (pybromo.Box): object defining the simulation box.
            rs (np.RandomState): numpy's RandomState used for initialization
                of the random number generator. If None, use a random state
                initialized from `seed`.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state. `seed` is ignored when `rs` is not None.
        """
        msg = 'The sequence `num_particles` must have length >= 1.'
        assert len(num_particles) > 0, msg
        msg = 'ERROR: `num_particles` and `D` must have the same length.'
        assert len(num_particles) == len(D), msg
        P = Particles(num_particles[0], D[0], box, rs=rs, seed=seed)
        for num_particle, D in zip(num_particles[1:], D[1:]):
            P.add(num_particles=num_particle, D=D)
        return P

    @staticmethod
    def num_particles_to_slices(num_particles_per_population):
        """
        Convert a list of number of particles per population
        into a list of `slice` object, each indexing a population.
        """
        slices = []
        i_prev = 0
        for num_particles in num_particles_per_population:
            slices.append(slice(i_prev, i_prev + num_particles))
            i_prev += num_particles
        return slices

    def __init__(self, num_particles, D, box, rs=None, seed=1, particles=None):
        """A set of `N` Particle() objects with random position in `box`.

        Arguments:
            num_particles (int): number of particles to be generated
            D (float): diffusion coefficient in S.I. units (m^2/s)
            box (Box object): the simulation box
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state. `seed` is ignored when `rs` is not None.
            particles (list or None): when not None, initialize the object from
                this list that must containing only `Particle` objects.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
        self.rs = rs
        self.init_random_state = rs.get_state()
        self.box = box
        if particles is None:
            self._plist = self._generate(num_particles, D, box, rs)
        else:
            self._plist = list(particles)
        self.rs_hash = hashfunc(self.init_random_state)[:3]

    def add(self, num_particles, D):
        """Add particles with diffusion coefficient `D` at random positions.
        """
        if D in self.diffusion_coeff:
            msg = ('A population with this diffusion coefficient is already '
                   'present. Change diffusion coefficient to add a new population.')
            raise ValueError(msg)
        self._plist += self._generate(num_particles, D, box=self.box, rs=self.rs)

    def to_list(self):
        return self._plist.copy()

    def to_json(self):
        return json.dumps({'particles': [v.to_dict() for v in self]})

    @classmethod
    def from_json(cls, json_str):
        particles = [Particle(**p) for p in json.loads(json_str)['particles']]
        # This returned obj will throw an error if the user calls .add()
        return cls(particles=particles, num_particles=None, D=None, box=None)

    def __iter__(self):
        return iter(self._plist)

    def __len__(self):
        return len(self._plist)

    def __getitem__(self, i):
        return self._plist[i]

    def __eq__(self, other_particles):
        if len(self) != len(other_particles):
            return False
        equal = np.array([p1 == p2 for p1, p2 in zip(self, other_particles)])
        return equal.all()

    @property
    def positions(self):
        """Initial position for each particle. Shape (N, 3, 1)."""
        return np.vstack([p.r0 for p in self]).reshape(len(self), 3, 1)

    @property
    def diffusion_coeff(self):
        return np.array([par.D for par in self])

    @property
    def num_populations(self):
        """Number of populations with different diffusion coefficient."""
        return len(self.particles_counts)

    @property
    def particles_counts(self):
        """Number of particles in each population."""
        return [c[1] for c in self.diffusion_coeff_counts]

    @property
    def diffusion_coeff_counts(self):
        """List of tuples of (diffusion coefficient, counts) pairs.

        The order of the diffusion coefficients is as in self.diffusion_coeff.
        """
        return [(diff_coeff, len(list(group)))
                for diff_coeff, group in itertools.groupby(self.diffusion_coeff)]

    def short_repr(self):
        s = ["P%d_D%.2g" % (n, D) for D, n in self.diffusion_coeff_counts]
        return "_".join(s)

    def __repr__(self):
        s = ["#Particles: %d D: %.2g" % (n, D)
             for D, n in self.diffusion_coeff_counts]
        return ", ".join(s)


def wrap_periodic(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that interval.
    This function is used to apply periodic boundary conditions.
    """
    a -= a1
    wrapped = np.mod(a, a2 - a1) + a1
    return wrapped


def wrap_mirror(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that interval.
    This function is used to apply mirror-like boundary conditions.
    """
    a[a > a2] = a2 - (a[a > a2] - a2)
    a[a < a1] = a1 + (a1 - a[a < a1])
    return a


class NoMatchError(Exception):
    pass


class MultipleMatchesError(Exception):
    pass


class ParticlesSimulation(object):
    """Class that performs the Brownian motion simulation of N particles.
    """
    _PREFIX_TRAJ = 'pybromo'
    _PREFIX_TS = 'times'

    @staticmethod
    def datafile_from_hash(hash_, prefix, path):
        """Return pathlib.Path for a data-file with given hash and prefix.
        """
        pattern = '%s_%s*.h*' % (prefix, hash_)
        datafiles = list(path.glob(pattern))
        if len(datafiles) == 0:
            raise NoMatchError('No matches for "%s"' % pattern)
        if len(datafiles) > 1:
            raise MultipleMatchesError('More than 1 match for "%s"' % pattern)
        return datafiles[0]

    @staticmethod
    def from_datafile(hash_, path='./', ignore_timestamps=False, mode='r'):
        """Load simulation from disk trajectories and (when present) timestamps.

        `mode` is only applied for opening a prexisting timestamp file.
        """
        path = Path(path)
        assert path.exists()

        file_traj = ParticlesSimulation.datafile_from_hash(
            hash_, prefix=ParticlesSimulation._PREFIX_TRAJ, path=path)
        store = TrajectoryStore(file_traj, mode='r')

        psf_pytables = store.h5file.get_node('/psf/default_psf')
        psf = psf_from_pytables(psf_pytables)
        box = store.h5file.get_node_attr('/parameters', 'box')
        P = store.h5file.get_node_attr('/parameters', 'particles')

        names = ['t_step', 't_max', 'EID', 'ID']
        kwargs = {name: store.numeric_params[name] for name in names}
        S = ParticlesSimulation(particles=Particles.from_json(P), box=box,
                                psf=psf, **kwargs)

        # Emulate S.open_store_traj()
        S.store = store
        S.psf_pytables = psf_pytables
        S.traj_group = S.store.h5file.root.trajectories
        S.emission = S.traj_group.emission
        S.emission_tot = S.traj_group.emission_tot
        if 'position' in S.traj_group:
            S.position = S.traj_group.position
        elif 'position_rz' in S.traj_group:
            S.position = S.traj_group.position_rz
        if not ignore_timestamps:
            try:
                file_ts = ParticlesSimulation.datafile_from_hash(
                    hash_, prefix=ParticlesSimulation._PREFIX_TS, path=path)
            except NoMatchError:
                # There are no timestamps saved.
                pass
            else:
                # Load the timestamps
                S.ts_store = TimestampStore(file_ts, mode=mode)
                S.ts_group = S.ts_store.h5file.root.timestamps
                print(' - Found matching timestamps.')
        return S

    @staticmethod
    def _get_group_randomstate(rs, seed, group):
        """Return a RandomState, equal to the input unless rs is None.

        When rs is None, try to get the random state from the
        'last_random_state' attribute in `group`. When not available,
        use `seed` to generate a random state. When seed is None the returned
        random state will have a random seed.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
            # Try to set the random state from the last session to preserve
            # a single random stream when simulating timestamps multiple times
            if 'last_random_state' in group._v_attrs:
                rs.set_state(group._v_attrs['last_random_state'])
                print("INFO: Random state set to last saved state in '%s'." %
                      group._v_name)
            else:
                print("INFO: Random state initialized from seed (%d)." % seed)
        return rs

    def __init__(self, t_step, t_max, particles, box, psf, EID=0, ID=0):
        """Initialize the simulation parameters.

        Arguments:
            D (float): diffusion coefficient (m/s^2)
            t_step (float): simulation time step (seconds)
            t_max (float): simulation time duration (seconds)
            particles (Particles object): initial particle position
            box (Box object): the simulation boundaries
            psf (GaussianPSF or NumericPSF object): the PSF used in simulation
            EID (int): index for the engine on which the simulation is ran.
                Used to distinguish simulations when using parallel computing.
            ID (int): an index for the simulation. Can be used to distinguish
                simulations that are run multiple times.

        Note that EID and ID are shown in the string representation and are
        used to save unique file names.
        """
        self.particles = particles
        self.box = box
        self.psf = psf
        self.t_step = t_step
        self.t_max = t_max
        self.ID = ID
        self.EID = EID
        self.n_samples = int(t_max / t_step)

    @property
    def diffusion_coeff(self):
        return self.particles.diffusion_coeff

    @property
    def num_particles(self):
        return len(self.particles)

    @property
    def sigma_1d(self):
        return [np.sqrt(2 * par.D * self.t_step) for par in self.particles]

    def __repr__(self):
        pM = self.concentration(pM=True)
        s = repr(self.box)
        s += "\n%s, %.1f pM, t_step %.1fus, t_max %.1fs" %\
             (self.particles, pM, self.t_step * 1e6, self.t_max)
        s += " ID_EID %d %d" % (self.ID, self.EID)
        return s

    def hash(self):
        """Return an hash for the simulation parameters (excluding ID and EID)
        This can be used to generate unique file names for simulations
        that have the same parameters and just different ID or EID.
        """
        hash_numeric = 't_step=%.3e, t_max=%.2f, np=%d, conc=%.2e' % \
            (self.t_step, self.t_max, self.num_particles, self.concentration())
        hash_list = [hash_numeric, self.particles.short_repr(), repr(self.box),
                     self.psf.hash()]
        return hashlib.md5(repr(hash_list).encode()).hexdigest()

    def compact_name_core(self, hashsize=6, t_max=False):
        """Compact representation of simulation params (no ID, EID and t_max)
        """
        Moles = self.concentration()
        name = "%s_%dpM_step%.1fus" % (
            self.particles.short_repr(), Moles * 1e12, self.t_step * 1e6)
        if hashsize > 0:
            name = self.hash()[:hashsize] + '_' + name
        if t_max:
            name += "_t_max%.1fs" % self.t_max
        return name

    def compact_name(self, hashsize=6):
        """Compact representation of all simulation parameters
        """
        # this can be made more robust for ID > 9 (double digit)
        s = self.compact_name_core(hashsize, t_max=True)
        s += "_ID%d-%d" % (self.ID, self.EID)
        return s

    @property
    def numeric_params(self):
        """A dict containing all the simulation numeric-parameters.

        The values are 2-element tuples: first element is the value and
        second element is a string describing the parameter (metadata).
        """
        nparams = dict(
            D = (self.diffusion_coeff.mean(), 'Diffusion coefficient (m^2/s)'),
            np = (self.num_particles, 'Number of simulated particles'),
            t_step = (self.t_step, 'Simulation time-step (s)'),
            t_max = (self.t_max, 'Simulation total time (s)'),
            ID = (self.ID, 'Simulation ID (int)'),
            EID = (self.EID, 'IPython Engine ID (int)'),
            pico_mol = (self.concentration() * 1e12,
                        'Particles concentration (pM)'))
        return nparams

    def print_sizes(self):
        """Print on-disk array sizes required for current set of parameters."""
        float_size = 4
        MB = 1024 * 1024
        size_ = self.n_samples * float_size
        em_size = size_ * self.num_particles / MB
        pos_size = 3 * size_ * self.num_particles / MB
        print("  Number of particles:", self.num_particles)
        print("  Number of time steps:", self.n_samples)
        print("  Emission array - 1 particle (float32): %.1f MB" % (size_ / MB))
        print("  Emission array (float32): %.1f MB" % em_size)
        print("  Position array (float32): %.1f MB " % pos_size)

    def concentration(self, pM=False):
        """Return the concentration (in Moles) of the particles in the box.
        """
        concentr = (self.num_particles / NA) / self.box.volume_L
        if pM:
            concentr *= 1e12
        return concentr

    __DOCS_STORE_ARGS___ = """
            prefix (string): file-name prefix for the HDF5 file.
            path (string): a folder where simulation data is saved.
            chunksize (int): chunk size used for the on-disk arrays saved
                during the brownian motion simulation. Does not apply to
                the timestamps arrays (see :method:``).
            chunkslice ('times' or 'bytes'): if 'bytes' (default) the chunksize
                is taken as the size in bytes of the chunks. Else, if 'times'
                chunksize is the size of the last dimension. In this latter
                case 2-D or 3-D arrays have bigger chunks than 1-D arrays.
            overwrite (bool): if True, overwrite the file if already exists.
                All the previously stored data in that file will be lost.
        """[1:]

    def _open_store(self, store, prefix='', path='./', mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Low level method used to implement different stores.

        Arguments:
            store (one of storage.Store classes): the store class to use.
        """ + self.__DOCS_STORE_ARGS___ + """
        Returns:
            Store object.
        """
        store_fname = '%s_%s.hdf5' % (prefix, self.compact_name())
        attr_params = dict(particles=self.particles.to_json(), box=self.box)
        kwargs = dict(path=path, nparams=self.numeric_params,
                      attr_params=attr_params, mode=mode)
        store_obj = store(store_fname, **kwargs)
        return store_obj

    def open_store_traj(self, path='./', chunksize=2**19, chunkslice='bytes',
                        mode='w', radial=False):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
        """ + self.__DOCS_STORE_ARGS___
        if hasattr(self, 'store'):
            return
        self.store = self._open_store(TrajectoryStore,
                                      prefix=ParticlesSimulation._PREFIX_TRAJ,
                                      path=path,
                                      mode=mode)

        self.psf_pytables = self.psf.to_hdf5(self.store.h5file, '/psf')
        self.store.h5file.create_hard_link('/psf', 'default_psf',
                                           target=self.psf_pytables)
        # Note psf.fname is the psf name in `h5file.root.psf`
        self.traj_group = self.store.h5file.root.trajectories
        self.traj_group._v_attrs['psf_name'] = self.psf.fname

        kwargs = dict(chunksize=chunksize, chunkslice=chunkslice)
        self.emission_tot = self.store.add_emission_tot(**kwargs)
        self.emission = self.store.add_emission(**kwargs)
        self.position = self.store.add_position(radial=radial, **kwargs)

    def open_store_timestamp(self, path=None, mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
        """ + self.__DOCS_STORE_ARGS___
        if hasattr(self, 'ts_store'):
            return
        if path is None:
            if hasattr(self, 'store'):
                # Use same folder of the trajectory file
                path = self.store.filepath.parent
            else:
                # No trajectory file, use current folder
                path = '.'
        self.ts_store = self._open_store(TimestampStore,
                                         prefix=ParticlesSimulation._PREFIX_TS,
                                         path=path,
                                         mode=mode)
        self.ts_group = self.ts_store.h5file.root.timestamps

    def _sim_trajectories(self, time_size, start_pos, rs,
                          total_emission=False, save_pos=False, radial=False,
                          wrap_func=wrap_periodic):
        """Simulate (in-memory) `time_size` steps of trajectories.

        Simulate Brownian motion diffusion and emission of all the particles.
        Uses the attributes: num_particles, sigma_1d, box, psf.

        Arguments:
            time_size (int): number of time steps to be simulated.
            start_pos (array): shape (num_particles, 3), particles start
                positions. This array is modified to store the end position
                after this method is called.
            rs (RandomState): a `numpy.random.RandomState` object used
                to generate the random numbers.
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            save_pos (bool): if True, save the particles 3D trajectories
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).

        Returns:
            POS (list): list of 3D trajectories arrays (3 x time_size)
            em (array): array of emission (total or per-particle)
        """
        time_size = int(time_size)
        num_particles = self.num_particles
        if total_emission:
            em = np.zeros(time_size, dtype=np.float32)
        else:
            em = np.zeros((num_particles, time_size), dtype=np.float32)

        POS = []
        # pos_w = np.zeros((3, c_size))
        for i, sigma_1d in enumerate(self.sigma_1d):
            delta_pos = rs.normal(loc=0, scale=sigma_1d,
                                  size=3 * time_size)
            delta_pos = delta_pos.reshape(3, time_size)
            pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
            pos += start_pos[i]

            # Coordinates wrapping using the specified boundary conditions
            for coord in (0, 1, 2):
                pos[coord] = wrap_func(pos[coord], *self.box.b[coord])

            # Sample the PSF along i-th trajectory then square to account
            # for emission and detection PSF.
            Ro = sqrt(pos[0]**2 + pos[1]**2)  # radial pos. on x-y plane
            Z = pos[2]
            current_em = self.psf.eval_xz(Ro, Z)**2
            if total_emission:
                # Add the current particle emission to the total emission
                em += current_em.astype(np.float32)
            else:
                # Store the individual emission of current particle
                em[i] = current_em.astype(np.float32)
            if save_pos:
                pos_save = np.vstack((Ro, Z)) if radial else pos
                POS.append(pos_save[np.newaxis, :, :])
            # Update start_pos in-place for current particle
            start_pos[i] = pos[:, -1:]
        return POS, em

    def _sim_trajectories_vfield(self, time_size, start_pos, rs,
                          total_emission=False, save_pos=False, radial=False,
                          wrap_func=wrap_periodic, vfield = [0.2, 0.2, 0.2]):
        """Simulate (in-memory) `time_size` steps of trajectories.

        Simulate Brownian motion diffusion and emission of all the particles.
        Uses the attributes: num_particles, sigma_1d, box, psf.

        Arguments:
            time_size (int): number of time steps to be simulated.
            start_pos (array): shape (num_particles, 3), particles start
                positions. This array is modified to store the end position
                after this method is called.
            rs (RandomState): a `numpy.random.RandomState` object used
                to generate the random numbers.
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            save_pos (bool): if True, save the particles 3D trajectories
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).

        Returns:
            POS (list): list of 3D trajectories arrays (3 x time_size)
            em (array): array of emission (total or per-particle)
        """
        deltat = self.t_step
        time_size = int(time_size)
        num_particles = self.num_particles
        if total_emission:
            em = np.zeros(time_size, dtype=np.float32)
        else:
            em = np.zeros((num_particles, time_size), dtype=np.float32)

        POS = []
        # pos_w = np.zeros((3, c_size))
        for i, sigma_1d in enumerate(self.sigma_1d):
            delta_pos = rs.normal(loc=0, scale=sigma_1d,
                                  size=3 * time_size)
            delta_pos = delta_pos.reshape(time_size, 3)
            delta_pos = (np.asarray(delta_pos) + np.asarray(vfield) * deltat).T
            pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
            pos += start_pos[i]

            # Coordinates wrapping using the specified boundary conditions
            for coord in (0, 1, 2):
                pos[coord] = wrap_func(pos[coord], *self.box.b[coord])

            # Sample the PSF along i-th trajectory then square to account
            # for emission and detection PSF.
            Ro = sqrt(pos[0]**2 + pos[1]**2)  # radial pos. on x-y plane
            Z = pos[2]
            current_em = self.psf.eval_xz(Ro, Z)**2
            if total_emission:
                # Add the current particle emission to the total emission
                em += current_em.astype(np.float32)
            else:
                # Store the individual emission of current particle
                em[i] = current_em.astype(np.float32)
            if save_pos:
                pos_save = np.vstack((Ro, Z)) if radial else pos
                POS.append(pos_save[np.newaxis, :, :])
            # Update start_pos in-place for current particle
            start_pos[i] = pos[:, -1:]
        return POS, em

    def simulate_diffusion(self, save_pos=False, total_emission=True,
                           radial=False, rs=None, seed=1, path='./',
                           wrap_func=wrap_periodic,
                           chunksize=2**19, chunkslice='times', verbose=True):
        """Simulate Brownian motion trajectories and emission rates.

        This method performs the Brownian motion simulation using the current
        set of parameters. Before running this method you can check the
        disk-space requirements using :method:`print_sizes`.

        Results are stored to disk in HDF5 format and are accessible in
        in `self.emission`, `self.emission_tot` and `self.position` as
        pytables arrays.

        Arguments:
            save_pos (bool): if True, save the particles 3D trajectories
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).
            path (string): a folder where simulation data is saved.
            verbose (bool): if False, prints no output.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
        self.open_store_traj(chunksize=chunksize, chunkslice=chunkslice,
                             radial=radial, path=path)
        # Save current random state for reproducibility
        self.traj_group._v_attrs['init_random_state'] = rs.get_state()

        em_store = self.emission_tot if total_emission else self.emission

        print('- Start trajectories simulation - %s' % ctime(), flush=True)
        if verbose:
            print('[PID %d] Diffusion time:' % os.getpid(), end='')
        i_chunk = 0
        t_chunk_size = self.emission.chunkshape[1]
        chunk_duration = t_chunk_size * self.t_step

        par_start_pos = self.particles.positions
        prev_time = 0
        for time_size in iter_chunksize(self.n_samples, t_chunk_size):
            if verbose:
                curr_time = int(chunk_duration * (i_chunk + 1))
                if curr_time > prev_time:
                    print(' %ds' % curr_time, end='', flush=True)
                    prev_time = curr_time

            POS, em = self._sim_trajectories(time_size, par_start_pos, rs,
                                             total_emission=total_emission,
                                             save_pos=save_pos, radial=radial,
                                             wrap_func=wrap_func)

            # Append em to the permanent storage
            # if total_emission, data is just a linear array
            # otherwise is a 2-D array (self.num_particles, c_size)
            em_store.append(em)
            if save_pos:
                self.position.append(np.vstack(POS).astype('float32'))
            i_chunk += 1
            self.store.h5file.flush()

        # Save current random state
        self.traj_group._v_attrs['last_random_state'] = rs.get_state()
        self.store.h5file.flush()
        print('\n- End trajectories simulation - %s' % ctime(), flush=True)

    def simulate_diff_plus_vfield(self, save_pos=False, total_emission=True,
                           radial=False, rs=None, seed=1, path='./',
                           wrap_func=wrap_periodic, chunksize=2**19,
                           chunkslice='times', verbose=True, vfield=[1,0,0]):
        """Simulate Brownian motion trajectories and emission rates in diff.

        This method performs the Brownian motion simulation using the current
        set of parameters. Before running this method you can check the
        disk-space requirements using :method:`print_sizes`.

        Results are stored to disk in HDF5 format and are accessible in
        in `self.emission`, `self.emission_tot` and `self.position` as
        pytables arrays.

        Arguments:
            save_pos (bool): if True, save the particles 3D trajectories
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).
            path (string): a folder where simulation data is saved.
            verbose (bool): if False, prints no output.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
        self.open_store_traj(chunksize=chunksize, chunkslice=chunkslice,
                             radial=radial, path=path)
        # Save current random state for reproducibility
        self.traj_group._v_attrs['init_random_state'] = rs.get_state()

        em_store = self.emission_tot if total_emission else self.emission

        print('- Start trajectories simulation - %s' % ctime(), flush=True)
        if verbose:
            print('[PID %d] Diffusion time:' % os.getpid(), end='')
        i_chunk = 0
        t_chunk_size = self.emission.chunkshape[1]
        chunk_duration = t_chunk_size * self.t_step

        par_start_pos = self.particles.positions
        prev_time = 0
        for time_size in iter_chunksize(self.n_samples, t_chunk_size):
            if verbose:
                curr_time = int(chunk_duration * (i_chunk + 1))
                if curr_time > prev_time:
                    print(' %ds' % curr_time, end='', flush=True)
                    prev_time = curr_time

            POS, em = self._sim_trajectories_vfield(time_size, par_start_pos,
                                        rs, total_emission=total_emission,
                                        save_pos=save_pos, radial=radial,
                                        wrap_func=wrap_func, vfield=vfield)

            # Append em to the permanent storage
            # if total_emission, data is just a linear array
            # otherwise is a 2-D array (self.num_particles, c_size)
            em_store.append(em)
            if save_pos:
                self.position.append(np.vstack(POS).astype('float32'))
            i_chunk += 1
            self.store.h5file.flush()
        # Save current random state
        self.traj_group._v_attrs['last_random_state'] = rs.get_state()
        self.store.h5file.flush()
        print('\n- End trajectories simulation - %s' % ctime(), flush=True)

    def _get_ts_name_mix_core(self, max_rates, populations, bg_rate,
                              timeslice=None):
        if timeslice is None:
            timeslice = self.t_max
        if populations is None:
            populations = [slice(0, self.num_particles)]
        s = []
        for ipop, (max_rate, pop) in enumerate(zip(max_rates, populations)):
            kw = dict(npop=ipop + 1, max_rate=max_rate,
                      npart=pop.stop - pop.start, pop=pop, bg_rate=bg_rate)
            s.append('Pop{npop}_P{npart}_Pstart{pop.start}_'
                     'max_rate{max_rate:.0f}cps_BG{bg_rate:.0f}cps'
                     .format(**kw))
        s.append('t_{}s'.format(timeslice))
        return '_'.join(s)

    def _get_ts_name_mix(self, max_rates, populations, bg_rate, rs,
                         hashsize=6):
        s = self._get_ts_name_mix_core(max_rates, populations, bg_rate)
        return '%s_rs_%s' % (s, hashfunc(rs.get_state())[:hashsize])

    def timestamps_match_pattern(self, pattern):
        return [t for t in self.timestamp_names if pattern in t]

    def timestamps_match_mix(self, max_rates, populations, bg_rate,
                             hash_=None):
        pattern = self._get_ts_name_mix_core(max_rates, populations, bg_rate)
        if hash_ is not None:
            pattern = '_'.join([pattern, 'rs', hash_])
        return self.timestamps_match_pattern(pattern)

    def get_timestamp_data(self, name):
        """Return matching (timestamps, particles, positions) pytables arrays.
        """
        par_name = name + '_par'
        pos_name = name + '_pos'
        timestamps = self.ts_store.h5file.get_node('/timestamps', name)
        particles = self.ts_store.h5file.get_node('/timestamps', par_name)
        try:
            positions = self.ts_store.h5file.get_node('/timestamps', pos_name)
        except tables.NoSuchNodeError:
            positions = None
        return timestamps, particles, positions

    @property
    def timestamp_names(self):
        names = []
        for node in self.ts_group._f_list_nodes():
            if node.name.endswith('_par') or node.name.endswith('_pos'):
                continue
            names.append(node.name)
        return names

    @staticmethod
    def _timestamps_from_counts(counts, time_axis, max_rate,
                                position=None, sort=True):
        """Compute timestamps from timetraces of counts.

        This function operates on a given "group" of particles
        (a population) and a given time chunk.
        Number of particles is `counts.shape[0]` and number of time bins is
        `counts.shape[1] == len(time_axis)`.

        The function takes a 2D array `counts` (1 row per particle,
        1 column per time bin), and a 1D array of times `time_axis`
        and generates an array of timestamps where counts > 0.
        When counts is > 1, there will be multiple identical timestamps
        (for example if counts is 3, there will be 3 identical timestamps).
        This function also computes particle number and (optionally)
        particle position for each timestamp.

        If `positions` is not None, returns also position of particles at each
        timestamps. `positions` should be an array of shape
        (num_particles, num_spatial_dims, num_time_bins) containing positions
        for the same particles and time bins in `counts`.

        Returns:
            A tuple of 3 arrays: timestamps, particles and positions.

        Notes:
            Particles number always starts at 0, as this function is unaware
            of the "real" particle ID. If needed, the caller should add an
            offset to the returned particles array to obtain
            the real particle ID.
        """
        if position is not None:
            # counts.shape[0] must be equal to position.shape[0]
            # or exceed it by 1 when it contains a background trace
            pos_part = position.shape[0]
            spatial_dims = position.shape[1]
            assert pos_part <= counts.shape[0] <= pos_part + 1
        max_counts = counts.max()
        if max_counts == 0:
            empty_pos = None
            if position is not None:
                empty_pos = np.empty(shape=(0, spatial_dims), dtype=np.float32)
            return (np.array([], dtype=np.int64),    # timestamps
                    np.array([], dtype=np.int64),    # particles
                    empty_pos)  # positions

        # These lists will contain one array per particle
        ts_times_parlist = []
        ts_particles_parlist = []
        ts_positions_parlist = []
        # Loop for each particle to compute timestamps
        for ip, counts_ip in enumerate(counts):
            # Compute timestamps for particle ip for all bins with counts > 0
            ts_times_by_num_counts = []
            ts_positions_by_num_counts = []
            for v in range(1, max_counts + 1):
                mask = counts_ip >= v
                ts_times_by_num_counts.append(time_axis[mask])
                if position is not None:
                    is_bg_particle = ip == position.shape[0]
                    if is_bg_particle:
                        shape = (mask.sum(), position.shape[1])
                        pos = np.full(shape, np.nan, dtype='float32')
                    else:
                        pos = position[ip, :, mask]
                    # list of 2D arrays
                    ts_positions_by_num_counts.append(pos)

            # Stack the timestamps from different "counts"
            ts = np.hstack(ts_times_by_num_counts)
            ts_times_parlist.append(ts)
            ts_particles_parlist.append(np.full(ts.size, ip, dtype='u1'))
            if position is not None:
                # concatenate 2D arrays by columns
                pos_current_particle = np.vstack(ts_positions_by_num_counts)
                ts_positions_parlist.append(pos_current_particle)

        # Merge the arrays of different particles
        ts_times = np.hstack(ts_times_parlist)
        ts_particles = np.hstack(ts_particles_parlist)
        # ts_positions are 2D, concatenate columns
        # (rows are spatial coordinates)
        ts_positions = None
        if position is not None:
            ts_positions = np.vstack(ts_positions_parlist)

        if sort:
            # Sort merged timestamps (from all particles)
            index_sort = ts_times.argsort(kind='mergesort')
            ts_times = ts_times[index_sort]
            ts_particles = ts_particles[index_sort]
            if position is not None:
                ts_positions = ts_positions[index_sort]

        return ts_times, ts_particles, ts_positions

    def _sim_timestamps_populations(self, emission, max_rates, populations,
                                    bg_rate, i_start, rs,
                                    position=None, scale=10):
        """Simulate timestamps for all the populations of particles.

        This method simulates timestamps for a time-chunk starting at
        the trajectory index `i_start` and ending at
        `i_start + emission.shape[1]`.

        Arguments:
            emission (array): 2D array of normalized emission rates
                (max emission is 1).
                Each row is a particle and each column a time step.
                This is emission is for a time-slice starting at `i_start`
                in the full trajectory.
            max_rates (list): list of max emission rates in Hz for each
                population.
            populations (list of 2-elemnt tuples): list of populations. Each
                population is define as a slice. For example,
                slice(4, 7) is a population with particles 4, 5, and 6.
                Particle IDs start at 0.
            bg_rate (float): rate of Poisson process simulating the background
            i_start (int): index in the full trajectory where the passed
                `emission` array starts.
            scale (int): factor to convert a time index to timestamps.
                For example, if a simulation has a time-step of
                500 nm, and scale = 10, the timestamps will increment in
                units of 50 ns.
            positions (None or array): array of shape
                `(num_particles, num_spatial_dims, num_time_bins)` containing
                particle positions for the same time chunk covered by
                the `emission` array.

        Returns:
            3 arrays for the current time-chunk:
            - `ts_times`: timestamps with unit `t_step / scale`
            - `ts_particles`: particle IDs for each timestamp
            - `ts_positions`: particle position for each timestamp
        """
        if populations is None:
            populations = [slice(0, self.num_particles)]
        save_pos = position is not None

        times = (i_start + np.arange(emission.shape[1], dtype='int64')) * scale

        # These lists will contain one array per population
        ts_times_poplist = []
        ts_particles_poplist = []
        ts_positions_poplist = []
        # Loop through populations
        for ipop, (max_rate, pop) in enumerate(zip(max_rates, populations)):
            is_last_population = ipop == len(populations) - 1
            bg = bg_rate if is_last_population else None
            emission_pop = emission[pop]
            position_pop = position[pop] if save_pos else None
            counts_pop = sim_counts_timetrace_with_bg(
                emission_pop, max_rate, bg, self.t_step, rs=rs)
            ts_times_pop, ts_particles_pop, ts_positions_pop = \
                self._timestamps_from_counts(
                    counts_pop, times, max_rate=max_rate,
                    sort=False, position=position_pop)
            ts_particles_pop += pop.start
            ts_times_poplist.append(ts_times_pop)
            ts_particles_poplist.append(ts_particles_pop)
            if save_pos:
                ts_positions_poplist.append(ts_positions_pop)

        # Merge populations
        ts_times = np.hstack(ts_times_poplist)
        ts_particles = np.hstack(ts_particles_poplist)
        ts_positions = None
        if save_pos:
            ts_positions = np.vstack(ts_positions_poplist)
            assert ts_positions.shape[0] == ts_times.shape[0]

        # Sort the merged timestamps (from all populations)
        index_sort = ts_times.argsort(kind='mergesort')
        ts_times = ts_times[index_sort]
        ts_particles = ts_particles[index_sort]
        if save_pos:
            ts_positions = ts_positions[index_sort]
        return ts_times, ts_particles, ts_positions

    def simulate_timestamps_mix(self, max_rates, populations, bg_rate,
                                rs=None, seed=1, chunksize=2**16,
                                comp_filter=None, overwrite=False,
                                skip_existing=False, scale=10, save_pos=False,
                                path=None, t_chunksize=None, timeslice=None):
        """Compute a timestamps array for a mixture of N populations.

        Timestamp data are saved to disk and accessible as pytables arrays in
        `._timestamps` and `._tparticles`.
        The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rates (list): list of the peak max emission rate for each
                population.
            populations (list of slices): slices to `self.particles`
                defining each population.
            bg_rate (float, cps): rate for a Poisson background process
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
                If None use default compression.
            overwrite (bool): if True, overwrite any pre-existing timestamps
                array. If False, never overwrite. The outcome of simulating an
                existing array is controlled by `skip_existing` flag.
            skip_existing (bool): if True, skip simulation if the same
                timestamps array is already present.
            scale (int): `self.t_step` is multiplied by `scale` to obtain the
                timestamps units in seconds.
            path (string): folder where to save the data.
            save_pos (bool): if True save 3D position of each particle
                for each emitted photon.
            timeslice (float or None): timestamps are simulated until
                `timeslice` seconds. If None, simulate until `self.t_max`.
        """
        self.open_store_timestamp(path=path)
        rs = self._get_group_randomstate(rs, seed, self.ts_group)
        if t_chunksize is None:
            t_chunksize = self.emission.chunkshape[1]
        timeslice_size = self.n_samples
        if timeslice is not None:
            timeslice_size = timeslice // self.t_step

        name = self._get_ts_name_mix(max_rates, populations, bg_rate, rs=rs)
        kw = dict(
            name=name, clk_p=self.t_step / scale,
            max_rates=max_rates, bg_rate=bg_rate, populations=populations,
            num_particles=self.num_particles,
            bg_particle=self.num_particles,
            overwrite=overwrite, chunksize=chunksize,
            save_pos=save_pos,
            )
        if save_pos:
            kw.update(spatial_dims=self.position.shape[1])
        if comp_filter is not None:
            kw.update(comp_filter=comp_filter)
        try:
            self._timestamps, self._tparticles, self._tpositions = (
                self.ts_store.add_timestamps(**kw))
        except ExistingArrayError as e:
            if skip_existing:
                print(' - Skipping already present timestamps array.')
                return
            else:
                raise e

        self.ts_group._v_attrs['init_random_state'] = rs.get_state()
        self._timestamps.attrs['init_random_state'] = rs.get_state()
        self._timestamps.attrs['PyBroMo'] = __version__

        ts_list, part_list, pos_list = [], [], []
        # Load emission in chunks, and save only the final timestamps
        prev_time = 0
        # Loop through time and for each time-slice simulate all populations
        pos_chunk = None
        for i_start, i_end in iter_chunk_index(timeslice_size, t_chunksize):

            curr_time = np.around(i_start * self.t_step, decimals=0)
            if curr_time > prev_time:
                print(' %.1fs' % curr_time, end='', flush=True)
                prev_time = curr_time

            em_chunk = self.emission[:, i_start:i_end]
            if save_pos:
                pos_chunk = self.position[:, :, i_start:i_end]

            ts_times_chunk, ts_particles_chunk, ts_positions_chunk = \
                self._sim_timestamps_populations(
                    em_chunk, max_rates, populations, bg_rate, i_start,
                    rs, scale=scale, position=pos_chunk)

            # Save sorted "photons" (suffix '_s')
            ts_list.append(ts_times_chunk)
            part_list.append(ts_particles_chunk)
            pos_list.append(ts_positions_chunk)  # it may be a list of None

        for ts, part, pos in zip(ts_list, part_list, pos_list):
            self._timestamps.append(ts)
            self._tparticles.append(part)
            if save_pos:
                self._tpositions.append(pos)

        # Save current random state so it can be resumed in the next session
        self.ts_group._v_attrs['last_random_state'] = rs.get_state()
        self._timestamps.attrs['last_random_state'] = rs.get_state()
        self.ts_store.h5file.flush()

    def simulate_timestamps_mix_da(self, max_rates_d, max_rates_a,
                                   populations, bg_rate_d, bg_rate_a,
                                   rs=None, seed=1, chunksize=2**16,
                                   comp_filter=None, overwrite=False,
                                   skip_existing=False, scale=10,
                                   path=None, t_chunksize=2**19,
                                   timeslice=None):

        """Compute D and A timestamps arrays for a mixture of N populations.

        This method reads the emission from disk once, and generates a pair
        of timestamps arrays (e.g. donor and acceptor) from each chunk.

        Timestamp data are saved to disk and accessible as pytables arrays in
        `._timestamps_d/a` and `._tparticles_d/a`.
        The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rates_d (list): list of the peak max emission rate in the
                donor channel for each population.
            max_rates_a (list): list of the peak max emission rate in the
                acceptor channel for each population.
            populations (list of slices): slices to `self.particles`
                defining each population.
            bg_rate_d (float, cps): rate for a Poisson background process
                in the donor channel.
            bg_rate_a (float, cps): rate for a Poisson background process
                in the acceptor channel.
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
                If None use default compression.
            overwrite (bool): if True, overwrite any pre-existing timestamps
                array. If False, never overwrite. The outcome of simulating an
                existing array is controlled by `skip_existing` flag.
            skip_existing (bool): if True, skip simulation if the same
                timestamps array is already present.
            scale (int): `self.t_step` is multiplied by `scale` to obtain the
                timestamps units in seconds.
            path (string): folder where to save the data.
            timeslice (float or None): timestamps are simulated until
                `timeslice` seconds. If None, simulate until `self.t_max`.
        """
        self.open_store_timestamp(path=path)
        rs = self._get_group_randomstate(rs, seed, self.ts_group)
        if t_chunksize is None:
            t_chunksize = self.emission.chunkshape[1]
        timeslice_size = self.n_samples
        if timeslice is not None:
            timeslice_size = timeslice // self.t_step

        name_d = self._get_ts_name_mix(max_rates_d, populations, bg_rate_d, rs)
        name_a = self._get_ts_name_mix(max_rates_a, populations, bg_rate_a, rs)

        kw = dict(clk_p=self.t_step / scale,
                  populations=populations,
                  num_particles=self.num_particles,
                  bg_particle=self.num_particles,
                  overwrite=overwrite, chunksize=chunksize)
        if comp_filter is not None:
            kw.update(comp_filter=comp_filter)

        kw.update(name=name_d, max_rates=max_rates_d, bg_rate=bg_rate_d)
        try:
            self._timestamps_d, self._tparticles_d, _ = (
                self.ts_store.add_timestamps(**kw))
        except ExistingArrayError as e:
            if skip_existing:
                print(' - Skipping already present timestamps array.')
                return
            else:
                raise e

        kw.update(name=name_a, max_rates=max_rates_a, bg_rate=bg_rate_a)
        try:
            self._timestamps_a, self._tparticles_a, _ = (
                self.ts_store.add_timestamps(**kw))
        except ExistingArrayError as e:
            if skip_existing:
                print(' - Skipping already present timestamps array.')
                return
            else:
                raise e

        self.ts_group._v_attrs['init_random_state'] = rs.get_state()
        self._timestamps_d.attrs['init_random_state'] = rs.get_state()
        self._timestamps_d.attrs['PyBroMo'] = __version__
        self._timestamps_a.attrs['init_random_state'] = rs.get_state()
        self._timestamps_a.attrs['PyBroMo'] = __version__

        # Load emission in chunks, and save only the final timestamps
        prev_time = 0
        for i_start, i_end in iter_chunk_index(timeslice_size, t_chunksize):

            curr_time = np.around(i_start * self.t_step, decimals=1)
            if curr_time > prev_time:
                print(' %.1fs' % curr_time, end='', flush=True)
                prev_time = curr_time

            em_chunk = self.emission[:, i_start:i_end]

            times_chunk_s_d, par_index_chunk_s_d, _ = \
                self._sim_timestamps_populations(
                    em_chunk, max_rates_d, populations, bg_rate_d, i_start,
                    rs=rs, scale=scale)

            times_chunk_s_a, par_index_chunk_s_a, _ = \
                self._sim_timestamps_populations(
                    em_chunk, max_rates_a, populations, bg_rate_a, i_start,
                    rs=rs, scale=scale)

            # Save sorted timestamps (suffix '_s') and corresponding particles
            self._timestamps_d.append(times_chunk_s_d)
            self._tparticles_d.append(par_index_chunk_s_d)
            self._timestamps_a.append(times_chunk_s_a)
            self._tparticles_a.append(par_index_chunk_s_a)

        # Save current random state so it can be resumed in the next session
        self.ts_group._v_attrs['last_random_state'] = rs.get_state()
        self._timestamps_d._v_attrs['last_random_state'] = rs.get_state()
        self.ts_store.h5file.flush()

    def simulate_timestamps_mix_da_online(self, max_rates_d, max_rates_a,
                                 populations, bg_rate_d, bg_rate_a,
                                 rs=None, seed=1, chunksize=2**16,
                                 comp_filter=None, overwrite=False,
                                 skip_existing=False, scale=10,
                                 path=None, t_chunksize=2**19,
                                 timeslice=None):
        """Compute D and A timestamps arrays for a mixture of N populations.

        This method simulates the diffusion, emission and generates a pair
        of timestamps arrays (e.g. donor and acceptor) all at the same time.
        This method avoids saving the trajectories to disk.

        Timestamp data are saved to disk and accessible as pytables arrays in
        `._timestamps_d/a` and `._tparticles_d/a`.
        The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rates (list): list of the peak max emission rate for each
                population.
            populations (list of slices): slices to `self.particles`
                defining each population.
            bg_rate (float, cps): rate for a Poisson background process
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
                If None use default compression.
            overwrite (bool): if True, overwrite any pre-existing timestamps
                array. If False, never overwrite. The outcome of simulating an
                existing array is controlled by `skip_existing` flag.
            skip_existing (bool): if True, skip simulation if the same
                timestamps array is already present.
            scale (int): `self.t_step` is multiplied by `scale` to obtain the
                timestamps units in seconds.
            path (string): folder where to save the data.
            timeslice (float or None): timestamps are simulated until
                `timeslice` seconds. If None, simulate until `self.t_max`.
        """
        self.open_store_timestamp(path=path)
        rs = self._get_group_randomstate(rs, seed, self.ts_group)
        if t_chunksize is None:
            t_chunksize = 2**19
        timeslice_size = self.n_samples
        if timeslice is not None:
            timeslice_size = timeslice // self.t_step

        name_d = self._get_ts_name_mix(max_rates_d, populations, bg_rate_d, rs)
        name_a = self._get_ts_name_mix(max_rates_a, populations, bg_rate_a, rs)

        kw = dict(clk_p=self.t_step / scale,
                  populations=populations,
                  num_particles=self.num_particles,
                  bg_particle=self.num_particles,
                  overwrite=overwrite, chunksize=chunksize)
        if comp_filter is not None:
            kw.update(comp_filter=comp_filter)

        kw.update(name=name_d, max_rates=max_rates_d, bg_rate=bg_rate_d)
        try:
            self._timestamps_d, self._tparticles_d, _ = (
                self.ts_store.add_timestamps(**kw))
        except ExistingArrayError as e:
            if skip_existing:
                print(' - Skipping, timestamps array already present.')
                return
            else:
                raise e

        kw.update(name=name_a, max_rates=max_rates_a, bg_rate=bg_rate_a)
        try:
            self._timestamps_a, self._tparticles_a, _ = (
                self.ts_store.add_timestamps(**kw))
        except ExistingArrayError as e:
            if skip_existing:
                print(' - Skipping, timestamps array already present.')
                return
            else:
                raise e

        self.ts_group._v_attrs['init_random_state'] = rs.get_state()
        self.ts_group.attrs['Diffusion'] = 1
        self._timestamps_d.attrs['init_random_state'] = rs.get_state()
        self._timestamps_d.attrs['PyBroMo'] = __version__
        self._timestamps_a.attrs['PyBroMo'] = __version__

        print('- Start trajectories simulation - %s' % ctime(), flush=True)
        par_start_pos = self.particles.positions

        # Load emission in chunks, and save only the final timestamps
        prev_time = 0
        for i_start, i_end in iter_chunk_index(timeslice_size, t_chunksize):

            curr_time = np.around(i_start * self.t_step, decimals=1)
            if curr_time > prev_time:
                print(' %.1fs' % curr_time, end='', flush=True)
                prev_time = curr_time

            _, em_chunk = self._sim_trajectories(t_chunksize, par_start_pos,
                                                 rs,
                                                 total_emission=False,
                                                 save_pos=False, radial=False,
                                                 wrap_func=wrap_periodic)

            times_chunk_s_d, par_index_chunk_s_d, _ = \
                self._sim_timestamps_populations(
                    em_chunk, max_rates_d, populations, bg_rate_d, i_start,
                    rs=rs, scale=scale)

            times_chunk_s_a, par_index_chunk_s_a, _ = \
                self._sim_timestamps_populations(
                    em_chunk, max_rates_a, populations, bg_rate_a, i_start,
                    rs=rs, scale=scale)

            # Save sorted timestamps (suffix '_s') and corresponding particles
            self._timestamps_d.append(times_chunk_s_d)
            self._tparticles_d.append(par_index_chunk_s_d)
            self._timestamps_a.append(times_chunk_s_a)
            self._tparticles_a.append(par_index_chunk_s_a)

        # Save current random state so it can be resumed in the next session
        self.ts_group._v_attrs['last_random_state'] = rs.get_state()
        self._timestamps_d._v_attrs['last_random_state'] = rs.get_state()
        self.ts_store.h5file.flush()
        print('\n- End trajectories simulation - %s' % ctime(), flush=True)


def sim_timetrace(emission, max_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).
    """
    emission_rates = emission * max_rate * t_step
    return np.random.poisson(lam=emission_rates).astype(np.uint8)


def sim_counts_timetrace_with_bg(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from r.v. ~ Poisson(emission_rates).

    Generate an array of counts on a binned time axis
    for one or more particles. Optionally, adds a trace for background counts.

    Arguments:
        emission (2D array): array of normalized emission rates. One row per
            particle (axis = 0). Columns are the different time steps.
        max_rate (float): the peak emission rate in Hz.
        bg_rate (float or None): rate of a constant Poisson background (Hz).
            Background is added as an additional row in the returned array
            of counts. If None, no background simulated.
        t_step (float): duration of a time step in seconds.
        rs (RandomState or None): object used to draw the random numbers.
            If None, a new RandomState is created using a random seed.

    Returns:
        `counts` an 2D uint8 array of counts in each time bin, for each
        particle. If `bg_rate` is None, then `counts.shape == emission.shape`.
        Otherwise, `counts` has one row more than `emission` for storing
        the background counts.
    """
    if rs is None:
        rs = np.random.RandomState()
    em = np.atleast_2d(emission).astype('float64', copy=False)
    counts_nrows = em.shape[0]
    if bg_rate is not None:
        counts_nrows += 1   # add a row for poisson background
    counts = np.zeros((counts_nrows, em.shape[1]), dtype='u1')
    # In-place computation
    # NOTE: the caller will see the modification
    em *= (max_rate * t_step)
    # Use automatic type conversion int64 (counts_par) -> uint8 (counts)
    counts_par = rs.poisson(lam=em)
    if bg_rate is None:
        counts[:] = counts_par
    else:
        counts[:-1] = counts_par
        counts[-1] = rs.poisson(lam=bg_rate * t_step, size=em.shape[1])
    return counts


def sim_timetrace_bg2(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from r.v. ~ Poisson(emission_rates).

    This is an alternative implementation of :func:`sim_timetrace_bg`.
    """
    if rs is None:
        rs = np.random.RandomState()
    emiss_bin_rate = np.zeros((emission.shape[0] + 1, emission.shape[1]),
                              dtype='float64')
    emiss_bin_rate[:-1] = emission * max_rate * t_step
    if bg_rate is not None:
        emiss_bin_rate[-1] = bg_rate * t_step
        counts = rs.poisson(lam=emiss_bin_rate).astype('uint8')
    else:
        counts = rs.poisson(lam=emiss_bin_rate[:-1]).astype('uint8')
    return counts
