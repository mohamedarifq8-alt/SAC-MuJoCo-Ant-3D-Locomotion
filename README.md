# 🚀 Mastering SAC: 3D Quadrupedal Locomotion (Ant-v4)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-112C3E)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-Production_RL-4B8BBE)](https://stable-baselines3.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

This repository documents the final and most advanced phase of the Reinforcement Learning journey: training a 4-legged, 8-joint robot (Ant) to navigate a 3D environment. To solve this complex high-dimensional continuous control task, the **Soft Actor-Critic (SAC)** algorithm was employed due to its exceptional sample efficiency.

---


<img width="350" height="300" alt="images (2)" src="https://github.com/user-attachments/assets/43305402-c651-4271-b231-de04a523382e" />
<img width="350" height="300" alt="Screenshot 1" src="https://github.com/user-attachments/assets/4412eab6-9393-4e00-9d49-0de7291b01ad" />


## 🛤️ The Journey (Project Phases)

Transitioning from 2D planes to 3D spaces required a deep understanding of complex physics and rigorous performance monitoring:

### Phase 1: 3D Environment Anatomy 
- **Description:** Deconstructing the physical properties and the 3D sensory inputs of the Ant.
- **Action Space:** 8-dimensional continuous space controlling the motors of all 4 legs (moving them forward, backward, up, and down) with torque limits of.
- **Observation Space:** 27-dimensional vector. To accurately represent 3D rotation (Roll, Pitch, Yaw), the environment feeds the neural network **Quaternions** rather than simple angles. Absolute positional coordinates were explicitly excluded to maintain Spatial Independence.
- **Termination:** The environment terminates if the Ant flips over (Z-height < 0.2m) or jumps erratically (Z-height > 1.0m).

### Phase 2: Complex Reward Engineering
- **Description:** Balancing speed with mechanical safety using a 4-part hybrid reward equation.
- **The Equation:** `Reward = Forward Velocity + Healthy Reward (+1) - Control Cost - Contact Cost`.
- **Techniques Used:** Unlike previous environments, the Ant is heavily penalized (`Contact Cost`) if it slams its mechanical feet too violently into the ground, forcing the SAC algorithm to discover an elegant and energy-efficient walking gait.

### Phase 3: Production Training & Log Monitoring
- **Description:** Training the agent for 1,000,000 steps and visualizing its learning progression.
- **Techniques Used:**
  * Utilizing an enormous Replay Buffer (`1,000,000`) and random warm-up steps (`learning_starts=10000`) for SAC.
  * Wrapping the environment with the `Monitor` wrapper to automatically log episode rewards, lengths, and times into a CSV file without manual intervention.
  * Designing a custom Python script to parse the `monitor.csv` and plot a **Rolling Average (window=100)**, effectively smoothing out the high-variance noise typical of SAC training to reveal the true learning breakthrough.

---

<img width="1188" height="586" alt="Screenshot 2" src="https://github.com/user-attachments/assets/e7fb9d5a-6a71-4d37-8352-25ae2673fa31" />

## 🧠 Core Concepts Explored

- **Soft Actor-Critic (SAC) in 3D:** Leveraging off-policy reinforcement learning to conquer 8 degrees of freedom in a 3D space.
- **Quaternions for AI:** Feeding continuous 3D rotational matrices (Quaternions) to Neural Networks instead of Euler angles to avoid gimbal lock.
- **Contact Costs:** Adding physical wear-and-tear constraints to the reward function to bridge the gap between simulation and real-world robotics (Sim2Real).
- **Rolling Average Visualization:** Transforming noisy step-by-step logs into smoothed, professional learning curves using Pandas and Seaborn.

---

## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
pip install torch numpy pandas matplotlib seaborn gymnasium[mujoco] stable-baselines3[extra] imageio
```

### Repository Structure

```text
📦 SAC-MuJoCo-Ant-3D-Locomotion
 ┣ 📜 "Ant withe SAC.py"              # Main script: Training, Monitor wrapping, and Plotting
 ┣ 📂 models                          
 ┃ ┗ 📜 sac_ant_part2.zip             # Final SB3 Production Model (1 Million+ Steps)
 ┣ 📂 videos                          
 ┃ ┗ 🎬 mujoco_ant_sac.mp4            # Rendered cinematic video of the Ant walking seamlessly
 ┗ 📜 README.md
```

---

## 🏋️ Training and Evaluation

To execute the training pipeline and generate the smoothed learning curves:
```bash
python "Ant withe SAC.py"
```

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*
