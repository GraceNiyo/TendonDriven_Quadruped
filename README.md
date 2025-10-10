# Quadruped Locomotion Control with Muscle Spindle Feedback

A MuJoCo-based simulation framework comparing different motor control strategies for quadruped locomotion with muscle spindle feedback.
This project investigates how different neural control mechanisms affect quadruped locomotion stability and performance. We compare four biologically-inspired motor control systems:
## Demo Video
[![Watch the demo](https://img.youtube.com/vi/4ji9BXXE7EQ/0.jpg)](https://youtu.be/4ji9BXXE7EQ)

### Control Systems Tested

| System | Description | Biological Basis |
|--------|-------------|------------------|
| **Feedforward** | Pure open-loop control without sensory feedback | Motor programs, learned movements |
| **Alpha-Gamma Collateral** | Gamma drive scaled by alpha drive via collaterals| alpha-to-gamma collaterals |
| **Independent Alpha-Gamma** | Separate neural pathways for muscle and spindle control | Theoretical independent control |
| **Beta Motor Neurons** | Single neurons control both muscle and spindle | Skeletofusimotor neurons |

### Key Features

- **Muscle Spindle Model**: Implements Ia and II afferent responses based on muscle length and velocity
- **Multiple Experimental Paradigms**: Full quadruped locomotion and single leg drop tests
- **Real-time Visualization**: Interactive MuJoCo viewer with camera controls
- **Comprehensive Data Collection**: Joint kinematics, muscle dynamics, ground forces, and sensory feedback

## Applications

- **Biomechanics Research**: Understanding natural locomotion control
- **Robotics**: Bio-inspired control for legged robots  
- **Neuroscience**: Motor control and sensorimotor integration

---
**Keywords**: quadruped locomotion, muscle spindle, motor control, biomechanics, MuJoCo, proprioception
