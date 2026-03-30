# Project 1 — Gaussian Process Surrogate for the Branin Function

## What this project is about

The central problem in engineering simulation is cost. A single 
CFD run can take hours, sometimes days. You cannot afford to run 
thousands of them during design optimization. Surrogate models 
solve this by learning the input-output relationship from a small 
number of carefully chosen simulation runs, then predicting the 
output everywhere else — instantly.

This project builds the simplest possible version of that workflow 
using a well-known mathematical benchmark called the Branin function.

## The Branin Function

The Branin function is a standard benchmark in surrogate modeling 
literature. It takes two inputs and produces one output: