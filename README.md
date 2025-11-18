# A Reinforcement Learning-Based Simulated Environment for Tactical Modeling in Offensive Football Scenarios

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PPO](https://img.shields.io/badge/RL-PPO-orange.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-API-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![RLlib](https://img.shields.io/badge/RLlib-Supported-orange)
![StableBaselines3](https://img.shields.io/badge/SB3-PPO-blue)
![Thesis](https://img.shields.io/badge/Academic-Thesis_Project-6f42c1)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)
![Football](https://img.shields.io/badge/Domain-Football_Analytics-brightgreen)
![RL](https://img.shields.io/badge/AI-Reinforcement_Learning-yellow)


> **Master’s Thesis Project — Politecnico di Torino**  
> This repository contains the official implementation of my master’s thesis in *Data Science & Engineering*.  
> The work focuses on designing a Reinforcement Learning environment for tactical football analysis, with agents learning offensive behaviours in realistic match scenarios.

## Overview

This repository contains a Reinforcement Learning framework developed to simulate tactical football scenarios on a two-dimensional 120×80 meter pitch. The system models attackers, defenders, and goalkeepers as autonomous agents that interact within a controlled environment, allowing the study of movement, decision-making, and spatial behaviour under realistic constraints.

At its core, the framework is built around several coordinated components. Players are represented through dedicated classes that manage roles, motion, field-of-view constraints, and ball possession. Ball behaviour is simulated with a physics module that governs trajectories and collisions. Reward functions rely on spatial grids and event-based incentives to guide the emergence of meaningful tactical patterns. A rendering module visualises the pitch, the reward structure, and agent trajectories, while structured training and evaluation pipelines handle learning procedures, logging, and reproducibility.

These components are implemented in a modular fashion to support systematic experimentation and allow the environment to integrate seamlessly with modern Reinforcement Learning libraries. The architecture is designed to evolve toward more complex settings, making it suitable for future extensions such as the incorporation of real match information derived from StatsBomb event data.

---

## Features

- **120×80 m football pitch** aligned with StatsBomb standards, with extended margins for normalization.
- **Single-agent and Multi-Agent environments**, including attackers, defenders, and a goalkeeper.
- **Physics-based ball mechanics** with possession logic, velocity, and collision handling.
- **Role-specific player classes** (Attacker, Defender, Goalkeeper) with configurable behaviour.
- **Reward shaping through spatial grids** and event-based incentives (shots, tackles, saves).
- **Multiple scenarios** (Move, Shot, View, Multi-Agent extensions).
- **Rendering** implementation of pitch, agents, trajectories, and optional reward heatmaps.
- **Modular architecture**, allowing custom scenarios, reward functions, and action spaces.
- **Full training pipeline** using PPO, with video rendering, logs, and reproducibility.

---

## Available Scenarios

### **Single-Agent Scenarios**
- **Move** – the agent learns to move toward the goal while optimizing spatial reward.
- **Shot** – simulates shooting mechanics with trajectory-based rewards.
- **View** – integrates field-of-view constraints restricting available actions.

### **Multi-Agent Scenario**
- **FootballMultiEnv** – attackers, defenders, and a goalkeeper interact simultaneously with independent observations, actions, and rewards. The environment supports multiple configurations (e.g., 1v1, 2v1, 2v2, 3v2) defined directly through the configuration files, allowing flexible setup of small-sided tactical scenarios.

---

## Installation

### 1. Clone the repository
Clone the project and move into its directory:

```
git clone https://github.com/Manuele23/Football-Tactical-AI.git
cd Football-Tactical-AI
```

### 2. Install the package
Install the project using the `pyproject.toml` configuration:

```
pip install -e .
```

## Quick Start

### 1. Test that installation works
Run a simple single-agent test episode (movement scenario):

```
python src/football_tactical_ai/env/test/testMove_SA.py
```

You should see a rendered animation of the agent moving on the pitch under:

```
test/videotest/testMove_SA.mp4
```


### 2. Train a model (example: Move scenario)
Launch the PPO training script:

```
python src/football_tactical_ai/env/training/trainingSingleAgent/trainingMoveSingleAgent.py
```

Training outputs (videos, logs, models) will appear under:

```
src/football_tactical_ai/env/training/
```

### 3. Evaluate the trained agent
After training, you can generate visualisations and metrics:

```
python src/football_tactical_ai/env/evaluation/eval_single_agent.py
```

This produces:
- videos under `evaluation/results/videos/`
- logs under `evaluation/results/logs/`

---

## Repository Structure

Below is a compact version of the repository structure, summarising the main modules and their organization.

```
FOOTBALL-Tactical-AI/
│
├── pyproject.toml
├── README.md
│
└── src/football_tactical_ai/
    ├── configs/                  # Config files (training, eval, pitch)
    ├── env/
    │   ├── objects/              # Ball, Pitch
    │   ├── scenarios/            # Single & multi-agent environments
    │   ├── evaluation/           # Eval scripts + logs & videos
    │   ├── helpers/              # Utility functions
    │   ├── players/              # Player classes (ATT, DEF, GK)
    │   ├── plots/                # Plot scripts for results
    │   └── training/             # Training pipelines, renders, rewards
    │
    ├── statsbomb360/             # StatsBomb analysis tools & role data
    └── test/                     # Test scripts & demos
```

---

## Academic Context

This work is part of my Master’s Thesis in *Data Science and Engineering* at the **Politecnico di Torino**, supervised by  
**Prof. Silvia Chiusano** and **MSc. Andrea Avignone**. The goal is to explore Reinforcement Learning techniques applied to football analytics and tactical modeling.

## License

This project is released under the **MIT License**, which allows unrestricted use, modification, and distribution of the code, provided that proper attribution to the original author is maintained.

## Contact

**Manuele Mustari**  
manuele.mustari@studenti.polito.it  
GitHub: https://github.com/Manuele23
