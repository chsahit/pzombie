# PZOMBIE: Panda with ZerO Mess taBletop IntEraction Simulator

## Setup

The only prerequisite is Drake. See their website for installation instructions.

## Overview

The idea here is to exploit Drake's realistic contact modeling and physics simulation without suffering from the typical learning curve. PZOMBIE provides a way to specify a scene an initial set of models and their poses and then simulate a policy acting in that scene. 


More formally, we define an `Env` in terms of `Asset` objects and their associated poses (class definitions in `components.py`). These poses are specified as 7 dimensional vectors with format (quaternion (real part first), x, y, z). A "policy" takes as input the current time and an object of type `State`, which provides the current object-centric state. The state object can also be queried for information about the current active contacts. As an output, the policy provides something of type `Action` to a low-level Cartesian stiffness controller. In its most general form, a `CartesianStiffnessAction` is a 6x6 Cartesian stiffness matrix and a setpoint for the gripper in joint space. There are also subclasses of this action for simpler stiffness commands (6-dimensional, axis aligned wrt the world frame), and for position commands which are essentially compliant motions that are stiff everywhere. 


Note that in practice, rather than returning something of type `Action` you may instead call the `InterpolationPolicy` also provided in `actions.py`. This sub-policy handles the logic of transforming a goal pose as an XYZ position of the gripper into a sequence of Actions that are evenly spaced across time. We also provide a module `kinematics.py` to help with IK/FK computations.


The control flow of PZOMBIE can be best understood by looking at some examples. You can run `examples/flip_cup.py` for a very simple example of a policy which grasps the cup, then picks it up, rotates it, and puts it down. More complicated than this, is `examples/eraser_demo.py`, which consists of moving to the eraser, picking it up, moving to a whiteboard, pressing down, and then executing a wipe motion.  In `examples/teleop_demo.py`, there is a simple proof-of-concept for teleoperating the Panda. The WASD keys control is motion in the xy-plane. The up and down arrow keys affect the Panda's motion in the z-axis, and the space bar actuates the gripper. Note that the terminal has to be "focused" in order for curses to pick up the keystrokes.
