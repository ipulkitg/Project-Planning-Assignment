# 🤖 Multi-Agent Logistics Planning with Reinforcement Learning

## 🧭 Overview

This project presents a custom **multi-agent logistics environment** designed to explore the use of **reinforcement learning (RL)** for efficient task planning and coordination in automated delivery systems. Inspired by the **crafting-world problem** and recent success in **Compositional Plan Vectors**, this project models real-world logistics operations involving **factories**, **recharge stations**, and **intelligent LogisticBots**.

Built using a **customized OpenAI Gym environment**, this simulation facilitates experimentation with several RL algorithms, including **DQN, SARSA, and A2C**, for sequential task planning in complex environments.

---

## 🧪 Motivation

In logistics, coordinating a sequence of dependent tasks—especially with multiple agents—is a critical yet challenging problem. Traditional algorithms often fall short when adapting dynamically to environmental constraints, energy limitations, and task dependencies. Our project addresses this challenge using RL for:

- Dynamic task scheduling
- Adaptive routing
- Efficient energy/resource management

---

## 🛠️ Key Components

### 🔁 Environments

- **Custom Logistic Environment**  
  Includes `Factory`, `RechargeStation`, and `LogisticBot` classes for a modular setup with defined reward mechanics.

- **Chaotic Warehouse Environment**  
  Realistic warehouse dynamics using the `stable-baselines3` framework and `Intrinsic Curiosity Module` to encourage exploration.

### 🧠 RL Algorithms Tested

- **DQN (Deep Q-Networks)** – ✅ Best performing
- **SARSA**
- **A2C (Advantage Actor Critic)**

### 📦 Agent Behavior

Agents are trained to:
- Pick up and deliver items across the grid
- Recharge efficiently to prevent power loss
- Avoid redundant or harmful actions (e.g., overcharging, failed pickups)
- Optimize delivery sequences

---

## 🎯 Reward Model

| Action                          | Reward/Penalty        |
|---------------------------------|------------------------|
| Visit important location        | +Small reward          |
| Successful delivery             | +Large reward          |
| Recharging (limited)            | +Scaled reward         |
| Redundant recharging            | -Penalty               |
| Full storage / failed pickup    | -Penalty               |
| Agent death                     | -Large penalty         |

---

## 📈 Results

DQN consistently outperformed other algorithms in custom and warehouse environments. Key metrics include:

- ⏱️ Delivery Time
- 🔋 Energy Consumption
- 🚚 Task Throughput

Experiments showed DQN-trained agents could dynamically sequence and execute multi-step tasks with high reliability, showcasing the potential of RL in autonomous logistics systems.

---

## 👨‍💻 Project Team

- **Pulkit Garg** – Team Lead  
- Aditya Chakravarthi  
- Ayush Joshi  
- Rupesh Barve  
- Aahn Sachin Deshpande  

---

## 📌 Conclusion

This project demonstrates how reinforcement learning, particularly DQN, can be applied to **complex, multi-agent logistics environments** to enable adaptive, efficient, and autonomous task planning. The results offer promising directions for improving **supply chain automation**, **robotics**, and **dynamic scheduling systems**.

---
## Demo
![output](https://github.com/user-attachments/assets/12f5a559-3058-4c2e-9315-b5c7be079647)

---
## 📎 Related Work

- [Wei et al., 2022] Compositional Plan Vectors in the Crafting World
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) – Reinforcement learning library
