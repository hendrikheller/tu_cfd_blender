__author__ = 'Hendrik Heller'

bl_info = {"name": "Tu CFD Import",
           "category": "Import-Export"}

import os
import time
import numpy as np
import bpy


class CfdImportPanel(bpy.types.Panel):
    """The UI Panel"""    # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "scene.cfdimportpanel"      # unique identifier for buttons and menu items to reference.
    bl_label = "Import CFD Data"       # display name in the interface.
    #bl_options = {'REGISTER', 'UNDO'}  # enable undo for the operator.
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        #scene = context.scene

        # Variablen
        layout.label(text=" Length between perpendiculars:")
        row = layout.row()
        row.prop(context.scene, "lpp")

        layout.label(text=" Reference speed:")
        row = layout.row()
        row.prop(context.scene, "u0")

        layout.label(text=" Carriage speed:")
        row = layout.row()
        row.prop(context.scene, "ucarriage")

        layout.label(text=" Writing steps of foout:")
        row = layout.row()
        row.prop(context.scene, "nfoout")

        layout.label(text=" Non-dimensional timestep:")
        row = layout.row()
        row.prop(context.scene, "dt")

        layout.label(text=" Filepaths:")
        row = layout.row()
        row.prop(context.scene, "path_state")
        row = layout.row()
        row.prop(context.scene, "path_foout")
        row = layout.row()
        row.prop(context.scene, "path_ship")

        row = layout.row()
        row.operator("scene.cfdimportoperator")


class CfdImportOperator(bpy.types.Operator):
    """Operator for Importing of CFD data"""      # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "scene.cfdimportoperator"        # unique identifier for buttons and menu items to reference.
    bl_label = "Import CFD Data"         # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # enable undo for the operator.

    def execute(self, context):        # execute() is called by blender when running the operator.

        lpp = bpy.context.scene.lpp
        u0 = bpy.context.scene.u0
        ucarriage = bpy.context.scene.ucarriage
        nfoout = bpy.context.scene.nfoout
        dt = bpy.context.scene.dt

        adj = nfoout * dt * (lpp / u0) * 24

        state = parse_state(bpy.context.scene.path_state)
        foout = parse_foout(bpy.context.scene.path_foout, lpp)

        create_animated_surface("surface_lin_0", foout, lin, 0, state, adj, bpy.context.scene.path_ship, lpp, nfoout, dt, u0, ucarriage)

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = state.length * dt * (lpp / u0) * 24

        return {'FINISHED'}            # this lets blender know the operator finished successfully.


def register():
    bpy.utils.register_class(CfdImportPanel)
    bpy.utils.register_class(CfdImportOperator)
    bpy.types.Scene.lpp = bpy.props.FloatProperty(name="lpp", description="Length between perpendiculars of the model.",
                                                  precision=4, default=6.0702)
    bpy.types.Scene.u0 = bpy.props.FloatProperty(name="u0", description="Reference speed of the model.", precision=4,
                                                 default=2.005)
    bpy.types.Scene.ucarriage = bpy.props.FloatProperty(name="ucarriage", description="Carriage speed.", precision=8,
                                                        default=1.86900085)
    bpy.types.Scene.nfoout = bpy.props.IntProperty(name="nfoout", description="Writing steps of foout.", default=100)
    bpy.types.Scene.dt = bpy.props.FloatProperty(name="dt", description="Non-dimensional timestep.", precision=11,
                                                 default=0.00330313015)
    bpy.types.Scene.path_state = bpy.props.StringProperty(name="state.dat", default=os.path.normpath("d:/blender kram/uhareksches ding/state_square.dat"), description="Define path to the state.dat file.", subtype='FILE_PATH')
    bpy.types.Scene.path_foout = bpy.props.StringProperty(name="foout.tec", default=os.path.normpath('d:/blender kram/uhareksches ding/foout_rect.dat'), description="Define path to the foout.tec file.", subtype='FILE_PATH')
    bpy.types.Scene.path_ship = bpy.props.StringProperty(name="ship.stl", default=os.path.normpath('d:/blender kram/uhareksches ding/KCSship duplex 6.stl'), description="Define path to the foout.tec file.", subtype='FILE_PATH')


def unregister():
    del bpy.types.Scene.lpp
    del bpy.types.Scene.u0
    del bpy.types.Scene.ucarriage
    del bpy.types.Scene.nfoout
    del bpy.types.Scene.dt
    del bpy.types.Scene.path_state
    del bpy.types.Scene.path_foout
    del bpy.types.Scene.path_ship
    bpy.utils.unregister_class(CfdImportPanel)
    bpy.utils.unregister_class(CfdImportOperator)

if __name__ == "__main__":
    register()


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
            dat[int(i/5)].extend(map(float, l))

    return StateDat(dat)


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
        self.length = len(self.data)

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


def parse_foout(path, lpp):
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
            data[step].append((l[0]*lpp, l[1]*lpp, -l[2]*lpp))
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


def insert_shapekey_keyframes(key, k, smoothing_function, smoothing_amount, adj):
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

        # temporary hard code
        #adj = nfoout * dt * (lpp / u0) * 24

        key.value = smoothing_function(s)
        fr = k+(i*adj)
        key.keyframe_insert("value", frame=fr)


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


def create_animated_surface(name, foout_data, smoothing_function, smoothing_amount, state_data, adj, path_ship,
                            lpp, nfoout, dt, u0, uschleppwagen):
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
    surface : Blender.object
        Handle for the created mesh.
    """
    surface = create_mesh(name, foout_data[0][0], calc_face_mapping(foout_data[1][0], foout_data[1][1]))
    #bpy.ops.shade_smooth(surface)
    surface.rotation_mode = 'ZYX'
    bpy.context.scene.objects.active = surface
    bpy.ops.object.shade_smooth()

    # import ship from .stl file
    bpy.ops.import_mesh.stl(filepath=path_ship)

    ship = bpy.context.scene.objects[path_ship.split(os.path.sep)[-1].split('.')[0]]
    ship.rotation_mode = 'ZYX'

    # create new camera if active scene has no camera attached
    if bpy.context.scene.camera is None:
        cam = bpy.data.cameras.new("Cam")
        cam_obj = bpy.data.objects.new("Cam", cam)
        bpy.context.scene.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
    else:
        cam_obj = bpy.context.scene.camera
    # position object
    cam_obj.location = (0, -3*lpp, 2*lpp)
    # rotate camera to look towards origin
    cam_obj.rotation_euler = ((50 / 180*np.pi), 0, 0)

    # Add Basis key
    surface.shape_key_add(from_mix=False)

    # Add a shape key for every step in foout
    for k, foout_step in enumerate(foout_data[0]):
        key = surface.shape_key_add("key_t" + str(k), from_mix=False)

        # calculate correct frame for k
        frame = k * nfoout * dt * (lpp / u0) * 24

        insert_shapekey_keyframes(key, frame, smoothing_function, smoothing_amount, adj)
        # todo: find a way to make the interpolation of the shape keys linear

        for i in range(len(key.data)):
            pt = key.data[i].co
            pt[0] = foout_data[0][k][i][0]
            pt[1] = foout_data[0][k][i][1]
            pt[2] = foout_data[0][k][i][2]

    for i in range(state_data.length):
        loc_x = state_data.get_step_var(i, 'xor') * lpp
        loc_y = state_data.get_step_var(i, 'yor') * lpp
        loc_z = state_data.get_step_var(i, 'zor') * lpp
        rot_phi = state_data.get_step_var(i, 'ang1')
        rot_theta = state_data.get_step_var(i, 'ang2')
        rot_psi = state_data.get_step_var(i, 'ang3')

        cur_time = state_data.get_step_var(i, 't') * dt * lpp / u0
        cam_x = cur_time * uschleppwagen + state_data.get_step_var(0, 'xor') * lpp

        cam_obj.location = (cam_x, -3*lpp, 2*lpp)

        frame = i * dt * (lpp / u0) * 24
        surface.location = (loc_x, loc_y, loc_z)
        surface.rotation_euler = (rot_phi, rot_theta, rot_psi)
        ship.location = (loc_x, loc_y, loc_z)
        ship.rotation_euler = (rot_phi, rot_theta, rot_psi)
        #set keyframes for animation
        surface.keyframe_insert(data_path="location", frame=frame)
        surface.keyframe_insert(data_path="rotation_euler", frame=frame)

        ship.keyframe_insert(data_path="location", frame=frame)
        ship.keyframe_insert(data_path="rotation_euler", frame=frame)

        if i == 0 or i == state_data.length-1:
            cam_obj.keyframe_insert(data_path="location", frame=frame, index=0)
            cam_obj.animation_data.action.fcurves[0].keyframe_points[-1].interpolation = 'LINEAR'

    return surface


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