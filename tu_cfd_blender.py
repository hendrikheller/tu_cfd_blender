__author__ = 'Hendrik Heller'

import os
import time
import numpy as np
import bpy


def clean_line(s):
    """Cleans a string of multiple whitespaces and line breaks

    Parameters
    ----------
    s : string
        The string to be cleaned.

    Returns
    -------
    t : string
        A string void of double white spaces and line breaks.
    """
    assert isinstance(s, str)
    t = " ".join(s.rstrip().split())
    assert only_single_spaces(t)
    assert t.count("\n") == 0
    return t


def parse_state(path):
    """Parse a state.dat file.

    <long description goes here>

    Parameters
    ----------
    path : string
        The system path to the state.dat file.

    Returns
    -------
    dat : [[values], ...]
        A list of lists each containing one step of parsed data.
    """
    assert isinstance(path, str)
    # transform path for os specifics
    path = os.path.normpath(path)
    datfile = open(path)
    lines = datfile.readlines()
    datfile.close()
    dat = []

    for i, l in enumerate(lines):
        l = clean_line(l).split(" ")

        if i % 5 == 0:
            assert len(l) == 4
            # append new list, 1 entry int, 3 entries float
            dat.append([int(l[0]), float(l[1]), float(l[2]), float(l[3])])
        else:
            assert len(l) == 6
            # extend list with floats
            dat[i/5].extend(map(float, l))

    return dat


class StateDat:
    """A class to better store the results of parsing a state.dat file and make them more accessible.

    Attributes:
    -----------
    varmapping : dictionary
        Assignes every of the 28 parsed values an index. This is where the value is stored within the internal array."""
    varmapping = {"t": 0, "gra1": 1, "gra2": 2, "gra3": 3,
                  "uship": 4, "vship": 5, "wship": 6, "upkt": 7, "vpkt": 8, "wpkt": 9,
                  "ome1": 10, "ome2": 11, "ome3": 12, "ppkt": 13, "qpkt": 14, "rpkt": 15,
                  "ang1": 16, "ang2": 17, "ang3": 18, "ang1p": 19, "ang2p": 20, "ang3p": 21,
                  "xor": 22, "yor": 23, "zor": 24, "ufest": 25, "vfest": 26, "wfest": 27}

    def __init__(self, dat):
        """Initializes a new 'StateDat' object

        Parameters
        ----------
        dat : [[t, gra1, gra2, gra3, uship, ...], ...]
            Any structure an nd-array can be initialized with, containing the data in the predefined order.
        """
        self.data = np.array(dat)

    def get_step(self, step):
        """Returns the data of a specific step.

        long description goes here

        Parameters
        ----------
        step : int
            The timestep to be returned.

        Returns
        -------
        data[step] : [values]
            A list of the values of data at the specified timestep.
        """
        return self.data[step]

    def get_var(self, var):
        """Returns data for a specified variable.

        Tries to match 'var' to one of the 28 predefined variables. If matching is successfull, all data points for this
        variable is returned.

        Parameters
        ----------
        var : string
            A string containing the name of the variable of which the data is to be returned.

        Returns
        -------
        data[:, var] : [values]
            A list of all data points for a specific variable.
        """
        assert isinstance(var, str)
        var = var.lower()
        if var in self.varmapping:
            return self.data[:, self.varmapping[var]]
        else:
            raise Exception("Var could not be matched.")

    def get_step_var(self, step, var):
        """Returns the data of a specific step for a specific variable.

        Tries to match 'var' to one of the 28 predefined variables. If matching is successful, the data point for 'var'
        at the timestep 'step' is returned

        Parameters
        ----------
        step : int
            The timestep to be returned.
        var : string
            A string containing the name of the variable of which the data is to be returned.

        Returns
        -------
        data[step, var] : float
            The data point for 'var' at 'timestep'.
        """
        assert isinstance(var, str)
        assert isinstance(step, int)
        var = var.lower()
        if var in self.varmapping:
            return self.data[step, self.varmapping[var]]
        else:
            raise Exception("Var could not be matched.")


def parse_foout(path):
    """Parse a foout.dat file.

    <long description goes here>

    Parameters
    ----------
    path : string
        The system path to the foout.dat file.

    Returns
    -------
    data, dims : [[vertexes], ...], (i, j, k)
        List of lists of all vertex infos for steps in the foout.dat and the dimensions of each step.
    """
    assert isinstance(path, str)
    # transform path for os specifics
    path = os.path.normpath(path)
    fooutfile = open(path)
    lines = fooutfile.readlines()
    fooutfile.close()

    # parsing the dimensions of the mesh (I, J, K)
    dims = clean_line(lines[6]).split(',')
    dims = map(return_digits, dims[:-1])
    dims = list(map(int, dims))

    # first four lines is uninteresting one-time header
    del lines[:4]

    data = []
    step = 0
    # number of steps is unknown so we process one step at a time and delete processed nodes
    while len(lines) != 0:
        del lines[:5]
        data.append([])
        for i in range(dims[0]*dims[1]):
            l = list(map(float, clean_line(lines[i]).split(' ')))
            # z coordinates are inverted
            data[step].append((l[0], l[1], -l[2]))
        del lines[:dims[0]*dims[1]]
        step += 1

    return data, dims


def calc_face_mapping(dim_i, dim_j):
    """Calculates the face mapping of rectangular meshes

    The connections between vertexes within the rectangular meshes are implicit, but have to be specified. This function
    calculates such a mapping.

    Parameters
    ----------
    dim_i : int
        First dimension of the rectangular mesh.
    dim_j : int
        Second dimension of the rectangular mesh.

    Returns
    -------
    mapping : [(v0, v1, v2, v3)]
        List of tuples each containing four vertexes that together form a face.
    """
    assert isinstance(dim_i, int)
    assert isinstance(dim_j, int)
    assert dim_i > 0 and dim_j > 0
    # create the face mapping
    mapping = []
    for i in range(dim_i-1):
        for j in range(dim_j-1):
            v0 = i*dim_j+j
            v1 = i*dim_j+j+1
            v2 = i*dim_j+j+dim_i
            v3 = i*dim_j+j+dim_i+1
            # order here is important for proper faces
            mapping.append((v0, v2, v3, v1))

    assert len(mapping) == (dim_i-1)*(dim_j-1)
    return mapping


def return_digits(s):
    """Returns all digits within a string.

    Filters a string completely and returns only the occuring digits.

    Parameters
    ----------
    s : string
        The string to be filtered for digits.

    Returns
    -------
    s : string
        A string containing only the digits occuring in the original string.
    """
    assert isinstance(s, str)
    result = ""
    for c in s:
        if c.isdigit():
            result += c
    return result


def calc_steps(start, stop, amount):
    """Returns a list of steps between 'start' and 'stop'

    Parameters
    ----------
    start : int
        The starting point.
    stop : int
        The ending point.
    amount : int
        The amount of steps to be calculated.

    Returns
    -------
    ret : [step0, step1, ...]
        A list of equally spaced steps between 'start' and 'stop'.
    """
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(amount, int)
    ret = []
    for i in range(amount):
        ret.append(start + i*(float(abs(start - stop))/float(amount-1)))
    assert len(ret) == amount
    return ret


def insert_shapekey_keyframes(key, k, smoothing_function, smoothing_amount):
    """Creates keyframes for shapekeys

    <long description>

    Parameters
    ----------
    key : object.shape_key
        The starting point.
    stop : int
        The ending point.
    amount : int
        The amount of steps to be calculated.

    Returns
    -------
    ret : [step0, step1, ...]
        A list of equally spaced steps between 'start' and 'stop'.
    """
    steps = calc_steps(-1, 1, 3+(smoothing_amount*2))
    for i, s in enumerate(steps):

        # todo: account for distance between frames (nfoout * dt * (lpp / u0) * 24)

        key.value = smoothing_function(s)
        key.keyframe_insert("value", frame=k+i)


def create_mesh(name, vertex_data, face_data):
    """Creates a new mesh object.

    Creates a new mesh in the active Blender environment. For every entry in 'vertex_data'  a vertex is created at the
    specified coordinates. Faces are created according to the face mapping in 'face_data'.

    Parameters
    ----------
    name : string
        The name of the mesh object to be created.
    vertex_data : [(x, y, z), ...]
        List of tuples, each containing the values for the coordinate system as floats.
    face_data : [(v0, v1, v2, v3), ...]
        List of tuples each containing four vertexes that together form a face.

    Returns
    -------
    ob : Blender.object
        Handle for the created mesh.
    """
    mesh = bpy.data.meshes.new("mesh")
    ob = bpy.data.objects.new(name, mesh)
    ob.location = bpy.context.scene.cursor_location
    bpy.context.scene.objects.link(ob)
    mesh.from_pydata(vertex_data, [], face_data)
    mesh.update(calc_edges=True)
    # ob.shade_smooth()
    return ob


def create_animated_surface(name, foout_data, smoothing_function, smoothing_amount):
    """Creates a new mesh object animated using shapekeys.

    Creates a new mesh in the active Blender environment. For every entry in 'vertex_data'  a vertex is created at the
    specified coordinates. Faces are created according to the face mapping in 'face_data'.
    Additionaly the development of the surface is animated using shapekeys and the data of the steps parsed from the
    foout.dat file.

    Parameters
    ----------
    name : string
        The name of the mesh object to be created.
    foout_data : [[vertexes], ...], (i, j, k)
        List of lists of all vertex infos for steps in the foout.dat and the dimensions of each step.
    smoothing_function : f(x)
        The function used to calculate the keyframe values for the shapekeys. Should return 0 at f(-1) and f(1) and 1 at
        f(0).
    smoothing_factor : int
        Determines the amount by which shapekeys are blended together. 0 = no smoothing.

    Returns
    -------
    ob : Blender.object
        Handle for the created mesh.
    """
    ob = create_mesh(name, foout_data[0][0], calc_face_mapping(foout_data[1][0], foout_data[1][1]))

    # Add Basis key
    ob.shape_key_add(from_mix=False)

    # Add a shape key for every step in foout
    for k, foout_step in enumerate(da[0]):
        key = ob.shape_key_add("key_t" + str(k), from_mix=False)

        # calculate correct frame for k
        frame = k * nfoout * dt * (lpp / u0) * 24

        insert_shapekey_keyframes(key, frame, smoothing_function, smoothing_amount)

        for i in range(len(key.data)):
            pt = key.data[i].co
            pt[0] = da[0][k][i][0]
            pt[1] = da[0][k][i][1]
            pt[2] = da[0][k][i][2]
    return ob


def lin(x):
    """A linear function for shapekey smoothing.

    A linear function designed to return 0 for x=-1 and x=1 and to return 1 for x=0

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
    """
    return -1*abs(float(x)) + 1


def quad(x):
    """A quadratic function for shapekey smoothing.

    A quadratic function designed to return 0 for x=-1 and x=1 and to return 1 for x=0

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
    """
    return -(x*x)+1


# --- TEST UTILITIES ---

def only_single_spaces(s):
    assert isinstance(s, str)
    space = False
    for c in s:
        if c == " ":
            if space is False:
                space = True
            else:
                return False
        else:
            space = False
    return True


def current_milli_time():
    return int(round(time.time() * 1000))
# ----------------------

# this will parse a foout.dat file and create an animated mesh from it when executed within a Blender environment
t0 = current_milli_time()

path_state = "d:/blender kram/uhareksches ding/state_square.dat"
path_foout = "d:/blender kram/uhareksches ding/foout_rect.dat"

# length between perpendiculars (laenge zwischen den loten)
lpp = 6.0702

# reference speed
u0 = 2.005

# writing steps of foout
nfoout = 100

# non-dimensional timestep
dt = 0.00330313015

da = parse_foout(path_foout)

create_animated_surface("surface_quad_0", da, lin, 0)

t1 = current_milli_time()

print ("runtime in millis: " + str(t1-t0))