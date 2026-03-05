\# Mini Project 3: Nuclear Reactor Control



Author: Uzezi Olorunmola  

Course: Reinforcement Learning (Spring 2026)  

Date: February 24, 2026



---



\## Overview



This project implements temporal difference learning (SARSA and Q-learning) to control a nuclear reactor under noisy sensor conditions. The agent learns to balance power generation with safety by keeping the reactor in the productive temperature range while avoiding meltdown.



\*\*Key Challenge:\*\* The agent can't see the true reactor temperature μt - only noisy sensor readings zt ~ N(μt, σ²). This partial observability makes it a POMDP, but TD learning handles it surprisingly well!



---



\## Quick Results



-Low Noise (σ=2.0):\*\*

\- SARSA: 2605 avg return, 0% meltdown rate

\- Q-Learning: 2519 avg return, 0% meltdown rate



\-High Noise (σ=4.0):\*\*

\- SARSA: 2332 avg return (-10.5% drop)

\- Q-Learning: 2372 avg return (-5.8% drop)



Key Finding: Q-learning was MORE robust to sensor noise than SARSA! The off-policy learning filtered out some of the observation noise.



---



\## Repository Structure

```

rl-mini3-reactor/

├── docs/

│   └── mini3\_reactor\_report.md    # Full written report (answers all questions)

├── src/

│   ├── reactor\_env.py              # Environment implementation

│   ├── sarsa\_agent.py              # SARSA algorithm

│   └── qlearning\_agent.py          # Q-learning algorithm

├── results/

│   ├── final\_comparison.png        # 4-panel comparison plot

│   ├── noise\_sensitivity.png       # Bar chart

│   └── \*.npy                       # Saved Q-tables and returns

├── train\_sarsa.py                  # Train SARSA (σ=2.0)

├── train\_qlearning.py              # Train Q-learning (σ=2.0)

├── train\_both\_high\_noise.py        # Train both algorithms (σ=4.0)

└── visualize\_results.py            # Generate comparison plots

```



---



\## How to Run



\*\*Install dependencies:\*\*

```bash

pip install numpy matplotlib

```



\*\*Train SARSA (low noise):\*\*

```bash

python train\_sarsa.py

```



\*\*Train Q-learning (low noise):\*\*

```bash

python train\_qlearning.py

```



\*\*Train both algorithms with high noise:\*\*

```bash

python train\_both\_high\_noise.py

```



\*\*Generate comparison plots:\*\*

```bash

python visualize\_results.py

```



---



\## Results



\### Learning Curves



!\[Final Comparison](results/final\_comparison.png)



-Top panels: Learning curves show SARSA (blue) learns faster at low noise, but Q-learning (red) is more robust at high noise.



-Bottom panels: Q-table heatmaps reveal clear structure - yellow/green in sweet spot bins (states 8-12), dark blue in danger zones.



\### Noise Sensitivity



!\[Noise Sensitivity](results/noise\_sensitivity.png)



Q-learning degraded less when sensor noise nearly doubled (~6% drop vs ~10% for SARSA).



---



\## Key Learnings



\*\*SARSA (on-policy):\*\*

\- Faster convergence at low noise

\- More conservative (good for safety)

\- But noise-sensitive because it learns from actual (noisy) actions



\*\*Q-learning (off-policy):\*\*

\- Slower initial learning

\- More robust to sensor noise

\- The max operator filters out some observation uncertainty



\*\*Partial observability:\*\* Even though states weren't fully Markovian, both algorithms converged! TD learning is robust to some degree of state aliasing.



---



\## Full Report



See \[`docs/mini3\_reactor\_report.md`](docs/mini3\_reactor\_report.md) for complete answers to all assignment questions including:

\- Formal MDP formulation

\- Algorithm justification (on-policy vs off-policy)

\- Detailed experimental results

\- Q-function analysis



---



\## Tools I Used



Python 3.13, NumPy, Matplotlib  

Git/GitHub for version control



---



\## Academic Integrity



Solo project by Uzezi Olorunmola.



LLM assistance: ChatGPT Perplexity \& Claude  for environment debugging and algorithm implementation, code verification. All code personally verified and understood.



---



\*\*Mini Project 3 - Problem 1: Nuclear Reactor Control with Temporal Difference Learning\*\*

