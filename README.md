tu_cfd_blender
==============

Introduction
------------
tu_cfd_blender is an addon for the free and open-source 3D computer graphics software blender, designed to import and
animate experiments in computational fluid dynamics.
It is specifically designed for data produced by the shipbuilding department at the Technische Universität Berlin.



Usage
-----
###Installing the addon
####User Preferences

####Script file

###Where to find the addon

###Adjusting parameters

###Specifying filepaths

###Importing the data



Example
-------
tbd



Input data
----------
###foout.dat
1 - A file header of 4 lines:
TITLE     = "Insert your title here"
VARIABLES = "X"
"Y"
"Z"

2 - followed by a reoccuring data header of 5 lines:  
ZONE T="Rectangular zone"  
 STRANDID=0, SOLUTIONTIME=0  
 I=250, J=250, K=1, ZONETYPE=Ordered  
 DATAPACKING=POINT  
 DT=(SINGLE SINGLE SINGLE )

where 'I', 'J' and 'K' determine the dimensions of the defined surface.  
3 - Followed by I*J*K lines, each defining a vertex of the surface.  
e.g. -2.500000000E+00 -2.000000000E+00 0.000000000E+00

Parts 2 and 3 can be repeated indefinitely.

###state.dat
Repeating 5 lines containing data in a specific order.

line 1: zeitschritt, gra1, gra2, gra3  
line 2: uship, vship, wship, upkt, vpkt, wpkt  
line 3: ome1, ome2, ome3, ppkt, qpkt, rpkt  
line 4: ang1, ang2, ang3, ang1p, ang2p, ang3p  
line 5: Xor, Yor, Zor, Ufest, Vfest, Wfest  

This structure can be repeated indefinitely.

###Ship mesh file (optional)
Here a previously created mesh file can be specified to be used in the animation process.
.STL and .PLY files are applicable.
