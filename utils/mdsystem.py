import os
import os.path
import numpy as np
import numpy.random
import itertools
import copy
import inspect

import scipy
import scipy.special
import scipy.integrate

try:
    import openmm
    from openmm import unit
    from openmm import app
except ImportError:  # Openmm < 7.6
    from simtk import openmm
    from simtk import unit
    from simtk.openmm import app

from openmmtools.constants import kB

pi = np.pi

DEFAULT_EWALD_ERROR_TOLERANCE = 1.0e-5 # default Ewald error tolerance
DEFAULT_CUTOFF_DISTANCE = 9.0 * unit.angstroms # default cutoff distance
DEFAULT_SWITCH_WIDTH = 1.5 * unit.angstroms # default switch width
DEFAULT_H_MASS = 1.0 * unit.amu # default hydrogen mass


def _read_oemol(filename):
    """Retrieve a molecule from a file as an OpenEye OEMol.

    This will raise an exception if the OpenEye toolkit is not installed or licensed.

    Parameters
    ----------
    filename : str
        Filename to read from, either absolute path or in data directory.

    Returns
    -------
    molecule : openeye.oechem.OEMol
        The molecule
    """
    if not os.path.exists(filename):
        filename = get_data_filename(filename)

    from openeye import oechem
    ifs = oechem.oemolistream(filename)
    mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, mol)
    ifs.close()
    return mol

def unwrap_py2(func):
    """Unwrap a wrapped function.
    The function inspect.unwrap has been implemented only in Python 3.4. With
    Python 2, this works only for functions wrapped by wraps_py2().
    """
    unwrapped_func = func
    try:
        while True:
            unwrapped_func = unwrapped_func.__wrapped__
    except AttributeError:
        return unwrapped_func

def handle_kwargs(func, defaults, input_kwargs):
    """Override defaults with provided kwargs that appear in `func` signature.

    Parameters
    ----------
    func : function
        The function to which the resulting modified kwargs is to be fed
    defaults : dict
        The default kwargs.
    input_kwargs: dict
        Input kwargs, which should override default kwargs or be added to output kwargs
        if the key is present in the function signature.

    Returns
    -------
    kwargs : dict
        Dictionary of kwargs that appear in function signature.

    """
    # Get arguments that appear in function signature.
    args, _, _, kwarg_defaults, _, _, _ = inspect.getfullargspec(unwrap_py2(func))
    # Add defaults
    kwargs = { k : v for (k,v) in defaults.items() }
    # Override those that appear in args
    kwargs.update({ k : v for (k,v) in input_kwargs.items() if k in args })

    return kwargs

def in_openmm_units(quantity):
    """Strip the units from a openmm.unit.Quantity object after converting to natural OpenMM units

    Parameters
    ----------
    quantity : openmm.unit.Quantity
       The quantity to convert

    Returns
    -------
    unitless_quantity : float
       The quantity in natural OpenMM units, stripped of units.

    """

    unitless_quantity = quantity.in_unit_system(unit.md_unit_system)
    unitless_quantity /= unitless_quantity.unit
    return unitless_quantity


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in testsystems.

    In the source distribution, these files are in ``openmmtools/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).

    """

    from pkg_resources import resource_filename
    fn = resource_filename('openmmtools', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


def halton_sequence(p, n):
    """
    Halton deterministic sequence on [0,1].

    Parameters
    ----------
    p : int
       Prime number for sequence.
    n : int
       Sequence length to generate.

    Returns
    -------
    u : numpy.array of double
       Sequence on [0,1].

    Notes
    -----
    Code source: http://blue.math.buffalo.edu/sauer2py/
    More info: http://en.wikipedia.org/wiki/Halton_sequence

    Examples
    --------
    Generate some sequences with different prime number bases.
    >>> x = halton_sequence(2,100)
    >>> y = halton_sequence(3,100)
    >>> z = halton_sequence(5,100)

    """
    eps = np.finfo(np.double).eps
    # largest number of digits (adding one for halton_sequence(2,64) corner case)
    b = np.zeros(int(np.ceil(np.log(n) / np.log(p))) + 1)
    u = np.empty(n)
    for j in range(n):
        i = 0
        b[0] += 1                       # add one to current integer
        while b[i] > p - 1 + eps:           # this loop does carrying in base p
            b[i] = 0
            i = i + 1
            b[i] += 1
        u[j] = 0
        for k in range(len(b)):         # add up reversed digits
            u[j] += b[k] * p**-(k + 1)
    return u


def subrandom_particle_positions(nparticles, box_vectors, method='sobol'):
    """Generate a deterministic list of subrandom particle positions.

    Parameters
    ----------
    nparticles : int
        The number of particles.
    box_vectors : openmm.unit.Quantity of (3,3) with units compatible with nanometer
        Periodic box vectors in which particles should lie.
    method : str, optional, default='sobol'
        Method for creating subrandom sequence (one of 'halton' or 'sobol')

    Returns
    -------
    positions : openmm.unit.Quantity of (natoms,3) with units compatible with nanometer
        The particle positions.

    Examples
    --------
    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors)

    Use halton sequence:

    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors, method='halton')

    """
    # Create positions array.
    positions = unit.Quantity(np.zeros([nparticles, 3], np.float32), unit.nanometers)

    if method == 'halton':
        # Fill in each dimension.
        primes = [2, 3, 5]  # prime bases for Halton sequence
        for dim in range(3):
            x = halton_sequence(primes[dim], nparticles)
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x * l / l.unit, l.unit)

    elif method == 'sobol':
        # Generate Sobol' sequence.
        from openmmtools import sobol
        ivec = sobol.i4_sobol_generate(3, nparticles, 1)
        x = np.array(ivec, np.float32)
        for dim in range(3):
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x[dim, :] * l / l.unit, l.unit)

    else:
        raise Exception("method '%s' must be 'halton' or 'sobol'" % method)

    return positions


def build_lattice_cell():
    """Build a single (4 atom) unit cell of a FCC lattice, assuming a cell length
    of 1.0.

    Returns
    -------
    xyz : np.ndarray, shape=(4, 3), dtype=float
        Coordinates of each particle in cell
    """
    xyz = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]
    xyz = np.array(xyz)

    return xyz


def build_lattice(n_particles):
    """Build a FCC lattice with n_particles, where (n_particles / 4) must be a cubed integer.

    Parameters
    ----------
    n_particles : int
        How many particles.

    Returns
    -------
    xyz : np.ndarray, shape=(n_particles, 3), dtype=float
        Coordinates of each particle in box.  Each subcell is based on a unit-sized
        cell output by build_lattice_cell()
    n : int
        The number of cells along each direction.  Because each cell has unit
        length, `n` is also the total box length of the `n_particles` system.

    Notes
    -----
    Equations eyeballed from http://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
    """
    n = ((n_particles / 4.) ** (1 / 3.))

    if np.abs(n - np.round(n)) > 1E-10:
        raise(ValueError("Must input 4 m^3 particles for some integer m!"))
    else:
        n = int(np.round(n))

    xyz = []
    cell = build_lattice_cell()
    x, y, z = np.eye(3)
    for atom, (i, j, k) in enumerate(itertools.product(np.arange(n), repeat=3)):
        xi = cell + i * x + j * y + k * z
        xyz.append(xi)

    xyz = np.concatenate(xyz)

    return xyz, n


def generate_dummy_trajectory(xyz, box):
    """Convert xyz coordinates and box vectors into an MDTraj Trajectory (with Topology)."""
    try:
        import mdtraj as md
        import pandas as pd
    except ImportError as e:
        print("Error: generate_dummy_trajectory() requires mdtraj and pandas!")
        raise(e)

    n_atoms = len(xyz)
    data = []

    for i in range(n_atoms):
        data.append(dict(serial=i, name="H", element="H", resSeq=i + 1, resName="UNK", chainID=0))

    data = pd.DataFrame(data)
    unitcell_lengths = box * np.ones((1, 3))
    unitcell_angles = 90 * np.ones((1, 3))
    top = md.Topology.from_dataframe(data, np.zeros((0, 2), dtype='int'))
    traj = md.Trajectory(xyz, top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

    return traj

def construct_restraining_potential(particle_indices, K):
    """Make a CustomExternalForce that puts an origin-centered spring on the chosen particles"""

    # Add a restraining potential centered at the origin.
    energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
    energy_expression += 'K = %f;' % (K / (unit.kilojoules_per_mole / unit.nanometers ** 2))  # in OpenMM units
    force = openmm.CustomExternalForce(energy_expression)
    for particle_index in particle_indices:
        force.addParticle(particle_index, [])
    return force


#=============================================================================================
# Thermodynamic state description
#=============================================================================================

class ThermodynamicState(object):

    """Object describing a thermodynamic state obeying Boltzmann statistics.

    Examples
    --------

    Specify an NVT state for a water box at 298 K.

    >>> from openmmtools import testsystems
    >>> system_container = testsystems.WaterBox()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin)

    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    A barostat will be added to the system if none is attached.

    Notes
    -----

    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    ToDo
    ----

    * Implement a more fundamental ProbabilityState as a base class?
    * Implement pH.

    """

    def __init__(self, system=None, temperature=None, pressure=None):
        """Construct a thermodynamic state with given system and temperature.

        Parameters
        ----------

        system : openmm.System, optional, default=None
            System object describing the potential energy function for the system
        temperature : openmm.unit.Quantity compatible with 'kelvin', optional, default=None
            Temperature for a system with constant temperature
        pressure : openmm.unit.Quantity compatible with 'atmospheres', optional, default=None
            If not None, specifies the pressure for constant-pressure systems.


        """

        self.system = system
        self.temperature = temperature
        self.pressure = pressure

        return

#=============================================================================================
# Abstract base class for test systems
#=============================================================================================


class TestSystem(object):

    """Abstract base class for test systems, demonstrating how to implement a test system.

    Parameters
    ----------

    Attributes
    ----------
    system : openmm.System
        System object for the test system
    positions : list
        positions of test system
    topology : list
        topology of the test system

    Notes
    -----

    Unimplemented methods will default to the base class methods, which raise a NotImplementedException.

    Examples
    --------

    Create a test system.

    >>> testsystem = TestSystem()

    Retrieve a deep copy of the System object.

    >>> system = testsystem.system

    Retrieve a deep copy of the positions.

    >>> positions = testsystem.positions

    Retrieve a deep copy of the topology.

    >>> topology = testsystem.topology

    Serialize system and positions to XML (to aid in debugging).

    >>> (system_xml, positions_xml) = testsystem.serialize()

    """

    def __init__(self, **kwargs):
        """Abstract base class for test system.

        Parameters
        ----------

        """

        # Create an empty system object.
        self._system = openmm.System()

        # Store positions.
        self._positions = unit.Quantity(np.zeros([0, 3], float), unit.nanometers)

        # Empty topology.
        self._topology = app.Topology()
        # MDTraj Topology is built on demand.
        self._mdtraj_topology = None

        return

    @property
    def system(self):
        """The openmm.System object corresponding to the test system."""
        return self._system

    @system.setter
    def system(self, value):
        self._system = value

    @system.deleter
    def system(self):
        del self._system

    @property
    def positions(self):
        """The openmm.unit.Quantity object containing the particle positions, with units compatible with openmm.unit.nanometers."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    @positions.deleter
    def positions(self):
        del self._positions

    @property
    def topology(self):
        """The openmm.app.Topology object corresponding to the test system."""
        return self._topology

    @topology.setter
    def topology(self, value):
        self._topology = value
        self._mdtraj_topology = None

    @topology.deleter
    def topology(self):
        del self._topology

    @property
    def mdtraj_topology(self):
        """The mdtraj.Topology object corresponding to the test system (read-only)."""
        import mdtraj as md
        if self._mdtraj_topology is None:
            self._mdtraj_topology = md.Topology.from_openmm(self._topology)
        return self._mdtraj_topology

    @property
    def analytical_properties(self):
        """A list of available analytical properties, accessible via 'get_propertyname(thermodynamic_state)' calls."""
        return [method[4:] for method in dir(self) if (method[0:4] == 'get_')]

    def reduced_potential_expectation(self, state_sampled_from, state_evaluated_in):
        """Calculate the expected potential energy in state_sampled_from, divided by kB * T in state_evaluated_in.

        Notes
        -----

        This is not called get_reduced_potential_expectation because this function
        requires two, not one, inputs.
        """

        if hasattr(self, "get_potential_expectation"):
            U = self.get_potential_expectation(state_sampled_from)
            U_red = U / (kB * state_evaluated_in.temperature)
            return U_red
        else:
            raise AttributeError("Cannot return reduced potential energy because system lacks get_potential_expectation")

    def serialize(self):
        """Return the System and positions in serialized XML form.

        Returns
        -------

        system_xml : str
            Serialized XML form of System object.

        state_xml : str
            Serialized XML form of State object containing particle positions.

        """
        try:
            from openmm import XmlSerializer
        except ImportError:  # OpenMM < 7.6
            from simtk.openmm import XmlSerializer

        # Serialize System.
        system_xml = XmlSerializer.serialize(self._system)

        # Serialize positions via State.
        if self._system.getNumParticles() == 0:
            # Cannot serialize the State of a system with no particles.
            state_xml = None
        else:
            platform = openmm.Platform.getPlatformByName('Reference')
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
            context = openmm.Context(self._system, integrator, platform)
            context.setPositions(self._positions)
            state = context.getState(getPositions=True)
            del context, integrator
            state_xml = XmlSerializer.serialize(state)

        return (system_xml, state_xml)

    @property
    def name(self):
        """The name of the test system."""
        return self.__class__.__name__


class AmberExplicitSystem(TestSystem):

    """Create amber explicit solvent system..

    Parameters
    ----------
    top : amber prmtop file
    crd : amber prmcrd file
    constraints : optional, default=openmm.app.HBonds
    rigid_water : bool, optional, default=True
    nonbondedCutoff : Quantity, optional, default=9.0 * unit.angstroms
    use_dispersion_correction : bool, optional, default=True
        If True, the long-range disperson correction will be used.
    nonbondedMethod : openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.
    cutoff : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_CUTOFF_DISTANCE
        Cutoff distance
    switch_width : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_SWITCH_WIDTH
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    ewaldErrorTolerance : float, optional, default=DEFAULT_EWALD_ERROR_TOLERANCE
           The Ewald or PME tolerance.

    Examples
    --------

    >>> alanine = AlanineDipeptideExplicit()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(self, top, crd, plumed=None ,constraints=app.HBonds, rigid_water=True, nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE, use_dispersion_correction=True, nonbondedMethod=app.PME, hydrogenMass=None, switch_width=DEFAULT_SWITCH_WIDTH, ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE, **kwargs):

        TestSystem.__init__(self, **kwargs)

#        prmtop_filename = get_data_filename(prmtop)
#        crd_filename = get_data_filename(inpcrd)

        # Initialize system.
        prmtop = app.AmberPrmtopFile(top)
        inpcrd = app.AmberInpcrdFile(crd)
        system = prmtop.createSystem(constraints=constraints, nonbondedMethod=nonbondedMethod, rigidWater=rigid_water, nonbondedCutoff=nonbondedCutoff, hydrogenMass=hydrogenMass)

        # Extract topology
        self.topology = prmtop.topology

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

        # Read positions.
        positions = inpcrd.getPositions(asNumpy=True)

        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        if plumed:
            from openmmplumed import PlumedForce
            plumed_infile = open(plumed,'r')
            try:
                plumed_script = plumed_infile.read()
            finally:
                plumed_infile.close()
            plumedforce = PlumedForce(plumed_script)
            system.addForce(plumedforce)
            

        self.system, self.positions = system, positions


class GromacsExplicitSystem(TestSystem):

    """Create amber explicit solvent system..

    Parameters
    ----------
    top : amber prmtop file
    crd : amber prmcrd file
    constraints : optional, default=openmm.app.HBonds
    rigid_water : bool, optional, default=True
    nonbondedCutoff : Quantity, optional, default=9.0 * unit.angstroms
    use_dispersion_correction : bool, optional, default=True
        If True, the long-range disperson correction will be used.
    nonbondedMethod : openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.
    cutoff : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_CUTOFF_DISTANCE
        Cutoff distance
    switch_width : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_SWITCH_WIDTH
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    ewaldErrorTolerance : float, optional, default=DEFAULT_EWALD_ERROR_TOLERANCE
           The Ewald or PME tolerance.

    Examples
    --------

    >>> alanine = AlanineDipeptideExplicit()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(self, top, gro, plumed=None, constraints=app.HBonds, rigid_water=True, nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE, use_dispersion_correction=True, nonbondedMethod=app.PME, hydrogenMass=None, switch_width=DEFAULT_SWITCH_WIDTH, ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE, **kwargs):

        TestSystem.__init__(self, **kwargs)

#        prmtop_filename = get_data_filename(prmtop)
#        crd_filename = get_data_filename(inpcrd)

        # Initialize system.
        gmxgro = app.GromacsGroFile(gro)
        gmxtop = app.GromacsTopFile(top, periodicBoxVectors=gmxgro.getPeriodicBoxVectors())
        system = gmxtop.createSystem(constraints=constraints, nonbondedMethod=nonbondedMethod, rigidWater=rigid_water, nonbondedCutoff=nonbondedCutoff, hydrogenMass=hydrogenMass)

        # Extract topology
        self.topology = gmxtop.topology

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

        # Read positions.
        #inpcrd = app.AmberInpcrdFile(crd)
        positions = gmxgro.getPositions(asNumpy=True)

        # Set box vectors.
        #box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        #system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
        if plumed:
            from openmmplumed import PlumedForce
            plumed_infile = open(plumed,'r')
            try:
                plumed_script = plumed_infile.read()
            finally:
                plumed_infile.close()
            plumedforce = PlumedForce(plumed_script)
            system.addForce(plumedforce)

        self.system, self.positions = system, positions




