**Mission**\\
For robot enthusiasts who seek to streamline sim-to-real development, THIMBLE is a low-cost bipedal robot that democratizes Reinforcement Learning development. Built using Open Duck as a frame, THIMBLE demonstrates that reinforcement learning can be low-cost and still produce good quality results.

**Target User**\\
The target user for this product is primarily robotics enthusiasts, students, and cosplayers, though the methods developed in the project are designed to be carried into larger projects.

**User Stories**\\
What follows are a number of imagined user statements that outline the goals of this project.\\
As a Masters student, I want a robot that is cheap to make.\\
As a robotics researcher, I want an RL pipeline that is readable, up-to-date, and easily modular/modifiable.\\
As a robotics hobbyist, I want a robot that I can show off to my friends.\\
As a robotics enthusiast, I want a robot that I can push and it self-balances and can move over multiple types of terrain (hardwood floor, soft carpet, stone, etc.)\\
As a cosplayer, I want a little guy who can follow me around and be cute.\\

**Feasibility and Riskiest Assumptions**\\
Key assumptions about the project:\\
Assumption 1 – Simulation-to-robot function still works for IsaacSim and is not deprecated (meaning we would not have to remodel the robot). (If it is deprecated, we might as well design our own robot and build our own URDF and not use open duck at all).\\
Assumption 2 – IsaacSim will properly model real-world physics and allow for sim-to-real transfer\\
Assumption 3 – We can accomplish sim-to-real transfer within 12 weeks.\\
Assumption 4 – Open Duck will be simple to build.\\
*Pivot Criteria*\\
Open Duck is more incomplete than anticipated, requiring more design work\\
By week 6, the RL simulation is still not producing ideal results.\\
Three servos break in the joints.\\
Simulation-to-robot function doesn’t actually port the robot model from onshape into IsaacLab (maybe deprecated)\\
By Week 10, the transfer to the real system is still not producing promising results (e.g. reward function is causing continuous errors, no noticeable improvement in behavior, etc.)\\

**Tooling**

**Demo Sentence**

**Faculty that can help and how**

**Evaluation and Metric**

**Potential Harm**

**Related Work**
