# A Reinforcement Learning-Based Simulated Environment for Tactical Modeling in Offensive Football Scenarios

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

## Experimental Extensions

### **What-If Tactical Analysis**

This extension bridges the gap between historical match data and generative simulation. By parsing  **StatsBomb event data** , the environment is initialized using the exact coordinates of real-world professional matches.

* **Counterfactual Replay:** Allows users to inject a specific historical event (e.g., a pass-shot assist) into the simulator to observe how RL agents resolve the scenario compared to the original professional outcome.
* **Data-Driven Initialization:** Replaces random agent placement with realistic spatial distributions, ensuring that tactical evaluations are grounded in high-leverage match contexts.
* **Performance Benchmarking:** Provides tools to calculate trajectory divergence and success rates across multiple deterministic rollouts of a single historical event.

### **Adversarial Learning**

To ensure the robustness of offensive policies, this module introduces a competitive "minimax" dynamic between teams.

* **Dynamic Opposition:** Unlike static or rule-based defenders, adversarial agents learn to minimize the offensive team's reward signal, creating an evolving defensive pressure.
* **Policy Stress-Testing:** Forces attackers to discover passing lanes and spatial solutions that are resilient to high-pressure interventions and interception attempts.
* **Competitive Co-evolution:** Both offensive and defensive policies improve simultaneously, leading to the emergence of more sophisticated and realistic tactical behaviors.

### **Realistic Player Customization**

This extension recognizes that tactical success is contingent upon individual physical capacities. It transitions the environment from "identical agents" to a  **heterogeneous multi-agent system** .

* **Attribute-Based Motion:** Introduces customizable parameters for individual agents, such as maximum speed and shot power.
* **Technical Profiling:** Allows for the adjustment of player-specific skills, including passing accuracy and shooting power, which directly modify the environment's transition dynamics and physics engine.
* **Scouting Simulation:** Enables "What-If" testing of specific player profiles within a team, evaluating how a change in a single agent's physical attributes affects the overall success of a collective tactical sequence.

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
python Test/testMove_SA.py
```

You should see a rendered animation of the agent moving on the pitch under:

```
Test/videoTest/testMove_SA.mp4
```

### 2. Train a model (example: Move scenario)

Launch the PPO training script:

```
python src/football_tactical_ai/training/trainingMultiAgent/trainingMultiAgent.py
```

Training outputs (videos, logs, models) will appear under:

```
src/football_tactical_ai/training/
```

### 3. Evaluate the trained agent

After training, you can generate visualisations and metrics:

```
python src/football_tactical_ai/evaluation/eval_single_agent.py
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
├── src/football_tactical_ai/
│    ├── configs/                  # Config files (training, eval, pitch)
│    ├── env/
│    │   ├── objects/              # Ball, Pitch
│    │   └── scenarios/            # Single & multi-agent environments
│    ├── evaluation/           # Eval scripts + logs & videos
│    ├── helpers/              # Utility functions
│    ├── players/              # Player classes (ATT, DEF, GK)
│    ├── plots/                # Plot scripts for evaluation results
│    └── training/             # Training pipelines, renders, rewards
│
├── pyproject.toml
├── README.md
├── statsbomb360/             # StatsBomb analysis tools & role data
├── test/                     # Test scripts & demos
│    └── whatif.ipynb         # Main interface of the what-if implementation 

```

---

## Academic Context

This work is part of the course "Applied data science project" at the **Politecnico di Torino**, starting from the thesis work of **Manuele Mustari**, supervised by **Prof. Silvia Chiusano** and **MSc. Andrea Avignone**. The goal is to explore Reinforcement Learning techniques applied to football analytics and tactical modeling.

## License

This project is released under the **MIT License**, which allows unrestricted use, modification, and distribution of the code, provided that proper attribution to the original author is maintained.
