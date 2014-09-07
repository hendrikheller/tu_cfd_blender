__author__ = 'Hendrik Heller'

import os
import numpy as np
import bpy

path_state = "d:/blender kram/uhareksches ding/state_square.dat"
path_foout = "d:/blender kram/uhareksches ding/foout_rect.dat"


def clean_line(s):
    # eliminate duplicate spaces and whitespaces from lines
    assert isinstance(s, str)
    t = " ".join(s.rstrip().split())
    assert only_single_spaces(t)
    assert t.count("\n") == 0
    return t


def parse_state(path):
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
    """class documentation"""
    varmapping = {"t": 0, "gra1": 1, "gra2": 2, "gra3": 3,
                  "uship": 4, "vship": 5, "wship": 6, "upkt": 7, "vpkt": 8, "wpkt": 9,
                  "ome1": 10, "ome2": 11, "ome3": 12, "ppkt": 13, "qpkt": 14, "rpkt": 15,
                  "ang1": 16, "ang2": 17, "ang3": 18, "ang1p": 19, "ang2p": 20, "ang3p": 21,
                  "xor": 22, "yor": 23, "zor": 24, "ufest": 25, "vfest": 26, "wfest": 27}

    def __init__(self, dat):
        self.data = np.array(dat)

    def get_step(self, step):
        return self.data[step]

    def get_step_var(self, step, var):
        assert isinstance(var, str)
        assert isinstance(step, int)
        var = var.lower()
        if var in self.varmapping:
            return self.data[step, self.varmapping[var]]
        else:
            raise Exception("Var could not be matched.")

    def get_var(self, var):
        assert isinstance(var, str)
        var = var.lower()
        if var in self.varmapping:
            return self.data[:, self.varmapping[var]]
        else:
            raise Exception("Var could not be matched.")


def parse_foout(path):
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
            data[step].append((l[0], l[1], l[2]))
        del lines[:dims[0]*dims[1]]
        step += 1

    return data, dims


def calc_face_mapping(dim_i, dim_j):
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
    assert isinstance(s, str)
    result = ""
    for c in s:
        if c.isdigit():
            result += c
    return result


def calc_steps(start, stop, amount):
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(amount, int)
    ret = []
    for i in range(amount):
        ret.append(start + i*(float(abs(start - stop))/float(amount-1)))
    assert len(ret) == amount
    return ret


def insert_shapekey_keyframes(key, k, smoothing_amount, smoothing_function):
    steps = calc_steps(-1, 1, 3+(smoothing_amount*2))
    for i, s in enumerate(steps):
        key.value = smoothing_function(s)
        key.keyframe_insert("value", frame=k+i)


def create_mesh(name, vertex_data, face_data):
    mesh = bpy.data.meshes.new("mesh")
    object = bpy.data.objects.new(name, mesh)
    object.location = bpy.context.scene.cursor_location
    bpy.context.scene.objects.link(object)
    mesh.from_pydata(vertex_data, [], face_data)
    mesh.update(calc_edges=True)
    # object.shade_smooth()
    return object


def lin(x):
    return -1*abs(float(x)) + 1


def quad(x):
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

import time
current_milli_time = lambda: int(round(time.time() * 1000))
# ----------------------

# if False:
#     import matplotlib.pyplot as plt
#     plt.subplot(111)
#     plt.plot(filedat.get_var("uship"), label="uship")
#     plt.plot(filedat.get_var("vship"), label="vship")
#     plt.plot(filedat.get_var("ang3"), label="ang3")
#     plt.plot(filedat.get_var("ang3p"), label="ang3p")
#     plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#     plt.grid()
#     plt.show()


def create_animated_surface(name, foout_data, smoothing_func, smoothing_amount):
    ob = create_mesh(name, foout_data[0][0], calc_face_mapping(foout_data[1][0], foout_data[1][1]))

    # Add Basis key
    ob.shape_key_add(from_mix=False)

    # Add a shape key for every step in foout
    for k, foout_step in enumerate(da[0]):
        key = ob.shape_key_add("key_t" + str(k), from_mix=False)
        insert_shapekey_keyframes(key, k, smoothing_amount, smoothing_func)

        for i in range(len(key.data)):
            pt = key.data[i].co
            pt[0] = da[0][k][i][0]
            pt[1] = da[0][k][i][1]
            pt[2] = da[0][k][i][2]


t0 = current_milli_time()
da = parse_foout(path_foout)
#print (da[1][0], da[1][1])

mapping = calc_face_mapping(da[1][0], da[1][1])

create_animated_surface("surface_quad_0", da, quad, 0)
create_animated_surface("surface_quad_1", da, quad, 1)
create_animated_surface("surface_quad_2", da, quad, 2)


t1 = current_milli_time()

# ob = create_mesh("surface", da[0][0], mapping)


# Add Basis key
# ob.shape_key_add(from_mix=False)

# Add a shape key for every step in foout
# for k, foout_step in enumerate(da[0]):
#     key = ob.shape_key_add("key_t" + str(k), from_mix=False)
#     insert_shapekey_keyframes(key, k, 0, lin)
#
#     for i in range(len(key.data)):
#         pt = key.data[i].co
#         pt[0] = da[0][k][i][0]
#         pt[1] = da[0][k][i][1]
#         pt[2] = da[0][k][i][2]


print ("runtime in millis: " + str(t1-t0))
#print len(mapping)
