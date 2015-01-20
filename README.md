tu_cfd_blender
==============

Introduction
------------
tu_cfd_blender is an addon for the free and open-source 3D computer graphics software blender, designed to import and
animate experiments in computational fluid dynamics.<br>
It is specifically designed for data produced by the Faculty of Mechanical Engineering and Transport Systems at Technische Universität Berlin.



Usage
-----
###Installing the addon
There are two ways load the TU CFD Import addon into Blender:
####User Preferences
Navigate to File -> User Preferences -> Addons.<br>
Pressing "Install from File..." opens up a file explorer, navigate to the "tucfdblender.py" file and confirm.<br>
The addon is now listed under "Import-Export: TU CFD Import".<br>
Activate the addon by checking the checkbox on the right.<br>
Pressing "Save User Settings" saves the current state permanently.

####Script file
For a single use of the addon open the *Text Editor* in one of active Blender windows.<br>
Open the "tucfdblender.py" file and press "Run Script".

This method will not keep the addon installed.

###Where to find the addon
The UI panel of the addon can now be found in the *Properties* window under the register *Scene*.


####Interpolation Mode of Keyframes
The best results are achieved by switching to a linear interpolation for the created keyframes. The standard used by
Blender is a bezier interpolation, which has an impact on the animation as well as the rendering performance.

This can be adjusted in the *Graph Editor*. Select the created surface and the ship object then switch to the *Graph Editor*,
select all keyframes (Select -> Select All / A) and change the interpolation mode to Linear (Key -> Interpolation Mode -> Linear / T -> Linear).

####Action Sets
Action sets can be used to add the once created animation to other objects. This can be used to change a ship model or
add it later in the process without the need to repeat the whole data import.

In order to add the ship animation to another object open up the *Dope Sheet* then change the Mode from "Dope Sheet" to
"Action Editor". Select the new object and choose the appropriate action to be linked from the list at the bottom.



Input data
----------
###foout.dat
1 - A file header of 4 lines:<br>
TITLE     = "Insert your title here"<br>
VARIABLES = "X"<br>
"Y"<br>
"Z"

2 - followed by a reoccuring data header of 5 lines:<br>
ZONE T="Rectangular zone"<br>
STRANDID=0, SOLUTIONTIME=0<br>
I=250, J=250, K=1, ZONETYPE=Ordered<br>
DATAPACKING=POINT<br>
DT=(SINGLE SINGLE SINGLE)

where 'I', 'J' and 'K' determine the dimensions of the defined surface.<br>
3 - Followed by I*J*K lines, each defining a vertex of the surface.<br>
e.g. -2.500000000E+00 -2.000000000E+00 0.000000000E+00

Parts 2 and 3 can be repeated indefinitely.

###state.dat
Repeating 5 lines containing data in a specific order.

line 1: zeitschritt, gra1, gra2, gra3<br>
line 2: uship, vship, wship, upkt, vpkt, wpkt<br>
line 3: ome1, ome2, ome3, ppkt, qpkt, rpkt<br>
line 4: ang1, ang2, ang3, ang1p, ang2p, ang3p<br>
line 5: Xor, Yor, Zor, Ufest, Vfest, Wfest

This structure can be repeated indefinitely.

###Ship mesh file (optional)
Here a previously created mesh file can be specified to be used in the animation process.<br>
.STL and .PLY files are applicable.