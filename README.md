# THIMBLE_sim_to_real
THIMBLE is a bipedal robot, designed as an implementation of self-balancing problems that come with bipedal robotics. It uses the Open Duck Mini V2 as a physical robot base and trains a reinforcement learning model in IsaacSim to self balance and walk over randomized domains.

THIMBLE is an acronym:
-Tiny\
-Homunculus \
-Iteratively\
-Making\
-Blunders, but\
-Learning\
-Eventually

THIMBLE is the capstone project for my masters degree in electrical and computer engineering at Boston University. It is intended to be both rigorous and whimsical, with the Open Duck Mini V2 modeled after characters in Star Wars (namely BDX droids designed for Disney parks) but still retaining the inherent challenges of bipedal locomotion.\

The folders in this project correspond to the following:\
-thimble_v1_sim -- RL Simulation Attempt 1. [DEPRECATED] Ultimately a failure due to incompatible URDF file that was too different from the final robot to guarantee sim-to-real transfer. Remade the model and found it easier to create a new project with the new model references rather than to go through the files and change the names individually.\
-thimble_v2 -- up to date URDF export of Thimble, now with handmade simple collision geometry around the torso and the bottoms of the feet.\
-thimble_v2_sim -- RL Simulation Attempt 2, currently active.\

