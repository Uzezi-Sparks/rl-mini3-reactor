\# Mini Project 3: Nuclear Reactor Control with Temporal Difference Learning



Name: Uzezi Olorunmola  

Date: \[Current Date]  

Course: Reinforcement Learning (Spring 2026)



---



\## Overview



This project implements TD learning algorithms (SARSA and Q-learning) to control a nuclear reactor under noisy sensor conditions. The agent must balance power generation with safety, keeping the reactor in a productive temperature range while avoiding meltdown.



---



\## Question 1: MDP Formulation



\### State Space (S)



\*\*What the agent observes:\*\* Noisy sensor readings zt ~ N(μt, σ²)



The true reactivity μt (core temperature) is HIDDEN. The agent only sees zt which has sensor noise added to it.



\*\*Discretization:\*\* I binned the continuous observations into n=20 discrete states.

\- Observation range: \[0, 100] (μmin to μmax)

\- Bin width: 5 units each

\- State space: S = {0, 1, 2, ..., 19} (bin indices)



\*\*Why 20 bins?\*\* Balances precision vs. learning speed. Too few bins (n=5) loses information about where the reactor actually is. Too many bins (n=50) makes the Q-table huge and takes forever to learn. n=20 is the sweet spot.



\### Action Space (A)



\*\*Actions:\*\* A = {-2, -1, 0, +1, +2} (5 discrete actions)



\*\*What they mean:\*\*

\- a = +2: Insert 2 rod increments → Cool reactor down (suppress reactivity)

\- a = +1: Insert 1 rod increment → Cool slightly

\- a = 0: Do nothing → Let reactor dynamics play out

\- a = -1: Withdraw 1 rod → Heat reactor up

\- a = -2: Withdraw 2 rods → Heat reactor up more



\*\*Why ±2 range?\*\* Small enough to maintain fine control, large enough to respond quickly when needed. The reactor has natural drift so we need meaningful action strength.



\### Transition Function P(z'|z,a)



\*\*The challenge:\*\* This is NOT actually Markovian! Here's why:



\*\*True dynamics:\*\*

