from dataclasses import dataclass
from typing import List, Dict, Union, TypeVar, Mapping
import numpy as np

@dataclass
class Point:
    latency: float
    cost: float
        
    def __repr__(self):
        return f'(latency={self.latency}, cost={self.cost})'
    
    def __hash__(self):
        return hash((self.latency, self.cost))
    
def plot_points(ax, points, color=None, label=None):
        latencies = [p.latency for p in points]
        costs = [p.cost for p in points]
        ax.scatter(latencies, costs, color=color, label=label)

def compute_pareto_frontier_naive(all_points: List[Point], max_pts=None) -> List[Point]:
    pareto_frontier = []
    for idx, p1 in enumerate(all_points):
        dominated = False
        for idx2, p2 in enumerate(all_points):
            if idx == idx2:
                continue 

            if p2.latency <= p1.latency and p2.cost <= p1.cost:
                dominated = True 

        if not dominated:
            pareto_frontier.append(p1)
        
    if max_pts is not None and len(pareto_frontier) > max_pts:
        pareto_frontier = np.random.choice(pareto_frontier, max_pts, replace=False)
        
    return pareto_frontier

def to_numpy(points: List[Point]):
    return np.array([[p.latency, p.cost] for p in points])

# Faster than is_pareto_efficient_simple, but less readable.
def _is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
def compute_pareto_frontier_efficient(points: List[Point]):
    as_np = to_numpy(points)
    indices = _is_pareto_efficient(as_np, False)
    filtered_points = [points[index] for index in indices]
    return filtered_points

@dataclass
class Stage:
    name: str
    frontier: List[Point] 
        
    def __str__(self):
        return f'{self.name}'
    
    def __repr__(self):
        return f'Stage(name={self.name}, frontier={len(self.frontier)})'
    
    def __hash__(self):
        return hash(self.name)
    
    def plot_frontier(self, ax, color=None):
        return plot_points(ax, self.frontier, color=color, label=self.name)