Characterization of Temporal Coordination in Spiking Neural Networks

This repository contains the simulation and training code developed for my Master’s thesis:
“Characterization of temporal coordination emerging in recurrent spiking neural networks trained to different computational tasks” (University of Bremen, 2025).

Overview

The project investigates how spiking neural networks (SNNs), composed of leaky integrate-and-fire neurons, can support flexible signal routing and attention-dependent communication in biologically inspired settings. The focus is on understanding how temporal coordination mechanisms such as synchrony and coincidence detection contribute to selective information transfer between neuronal populations.

Two core network architectures are implemented:

AB model: a minimal sender–receiver setup, where recurrent coupling in the sender shapes coincidence detection in the receiver.
aAB model: an extended architecture with a control population (a) that modulates the sender, enabling attention-dependent selective routing.

Both models are studied with:

Fixed weights to reproduce hand-tuned dynamics of synchrony and routing.

Trainable weights optimized using backpropagation through time (BPTT) and surrogate gradients, exploring how routing can emerge via learning.
