---
layout: about
title: about
permalink: /
#subtitle: <a href='#'>Affiliations</a>. Address. Contacts. Moto. Etc.

profile:
  align: left
  image: mountains.jpg
  image_circular: false # crops the image to make it circular
  address: >
    <p>contact me:</p>
    <p>ilze.auzina@bethgelab.org</p>

news: false  # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page
---

## About me 
I am an [ELLIS PhD](https://ellis.eu/phd-postdoc) researcher in Machine Learning working with [Prof. Dr. Matthias Bethge](https://scholar.google.de/citations?user=0z0fNxUAAAAJ&hl=en) at Tübingen University, Germany. Previously, I worked with [Dr. Efstratios Gavves](https://www.egavves.com/) and [Dr. Sara Magliacane](https://saramagliacane.github.io/) at University of Amsterdam in [Video & Image Sense Lab](https://ivi.fnwi.uva.nl/vislab/). 

Before starting my PhD I collaborated with [Dr. Jakub M. Tomczak](https://jmtomczak.github.io/) on my Master Thesis at Vrije Universiteit, Amsterdam on [Approximate Bayesian Computation for discrete data](https://www.mdpi.com/1099-4300/23/3/312), which was nominated for best Master Thesis award. 

Currently, my research is driven by a central question:

_How can we successfully extend RL post-training beyond math and coding tasks to open-ended problems?_


---------------------

### Research Focus 

#### ML for Science & Dynamical Systems (Early PhD)

In the early stages of my PhD, I explored machine learning for scientific modeling, particularly dynamical systems. I focused on teaching neural networks to infer latent dynamics from high-dimensional data while enforcing appropriate structural or physical constraints.

This included work on:
- Neural ODEs and GP-ODEs combined with VAE-based latent dynamics
- Modulated dynamics with explicit parameter control
- Ensuring learned systems reflect true physical structure rather than overfitting observations

-------

#### Shift Toward RL Post-Training

As language models rapidly advanced, my work shifted toward reinforcement learning post-training — the stage where large models are shaped using reward models, preferences, and multi-step decision pipelines.

I currently work on:
- Dense reward modeling and objective shaping (e.g., LoRA-based reward models, RLVN-style approaches)
- Stability in RLHF pipelines, including normalization, clipping, and reward-scaling strategies
- Open-ended and multi-step environments where reward sparsity requires principled training design

Incentive-driven behaviors: how reward models influence the reasoning and decision patterns of LLMs

My goal is to make RL post-training more stable, interpretable, and sample-efficient, and to understand how reinforcement signals interact with large neural models.

-------

### What Drives Me

I enjoy research that combines:
- Real-world impact: Agents will play an increasingly central role in how we work and interact with technology. We should design them in ways that genuinely improve human workflows, reasoning, and decision-making.
- Practical design: I value methods that are simple, robust, and usable - research that others can build on, deploy, or learn from without unnecessary complexity.
- Deep understanding: I’m motivated by foundational questions: Is RL genuinely enabling novel reasoning abilities, or is it primarily sharpening what supervised training already provides? Answering questions like these helps us design better, more grounded learning systems.

Ultimately, I aim to work on problems at the frontier — areas that matter but are often overlooked or underexplored in mainstream industry research. If our research interests intersect, feel free to reach out. I’m always happy to connect, collaborate, and discuss ideas.
