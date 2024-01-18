# Brief overview 

This repository contains the code necessary to reproduce the work of my Bachelor Thesis, which aims to extend the work of [Fei Song, Khaled Zaouk, Chenghao Lyu, Arnab Sinha, Qi Fan, Yanlei Diao, and Prashant Shenoy.](https://ieeexplore.ieee.org/document/9458826). In short, we introduce a novel algorithm to compute the Pareto Frontier for computational graphs with multiple optimization objectives. This is particularly relevant for compute providers with large clusters, such as AWS or Alibabe Cloud. Please refer to the Abstract and Introduction of [the paper](paper.pdf). 

## How to read this code 

The [demo file](demo.ipynb) contains everything that is needed to reproduce the paper. [base.py](base.py) contains functionality for efficiently computing Pareto Frontiers, while [graphs.py](graphs.py) contains helper classes for the Divide-And-Conquer algorithm presented in the paper

## Motivating example 

The following is simply an extract from Section 2.1 of [the paper](paper.pdf), it is pasted here to clarify the motivation.

Let us imagine that we have a very long-running task that we want to execute on our local cluster (that is also busy with other tasks) with Apache Spark. For example, we have a list of millions of books that we store on a local hard drive and we want to make sure that their contents are equivalent to what can be found online. Note that this is a toy example that we use to illustrate our problem and hence many simplifications will be made. To execute this task we break it down to the following steps:

1. Fetch the online versions of the books from some website.
2. Fetch our own versions from our hard drive.
3. Compute hashes of the online books’ content.
4. Compute hashes of the local books’ content.
5. Compare the hashes and return any possible differences.

Observe that we have some requirements on the order of executions of the above steps. For instance, steps 1 and 2 can be executed in parallel, step 3 has to follow step 1, step 4 has to follow step 2, and step 5 has to come last. These complex dependencies can be encoded by a dependency graph as shown in Figure 1. This dependency graph is how our Spark query will look like. After feeding it to the UDAO system (from [7]) we will get different configurations and resulting objectives for every node in this graph. For simplicity, let us assume that UDAO returned two configurations for every node such that:

- Configuration 1 is one that provides the lowest possible latency but requires a lot of resources and hence the cost of running it on our cluster is high.
- Configuration 2 is one that provides the highest possible latency but requires little resources and hence the cost of running it on our cluster is low.

Even in this simple example, there are many configurations that one can choose for our task. For example, we could:

- Execute both fetches fast with high cost, then execute both hash computations slowly with low cost, then execute the comparison fast.
- Or execute everything slowly with low cost.
- Or execute the local part of the graph fast and the rest slowly.
- Or...

However, out of all these configurations, only a few make sense. Indeed, one can imagine that the "Fetch Online" node will have much higher latencies than the rest of the nodes simply because it needs to download millions of books from the internet. In this context, we know that no matter how slowly we execute the "Fetch Local + Compute Hashes 2" portion of the graph, the bottleneck of the computation will always be the "Fetch Online" node. As such, it makes sense to allocate a lot of resources to "Fetch online + Compute Hashes 1" (so choose Configuration 1 for these two nodes) while choosing Configuration 2 for nodes "Fetch Local + Compute Hashes 2". The configuration for "Compare Hashes" could be anything.

This example illustrates exactly the task we will be trying to solve in this paper. Given a Spark Query and a Pareto optimal list of configurations for each node (i.e., each stage), find configurations that are optimal for the whole graph. We call the list of all optimal configurations for a graph its Pareto Frontier.
