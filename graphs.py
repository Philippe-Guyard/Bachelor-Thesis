import time 
import string 
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import itertools

from dataclasses import dataclass, field
from typing import List, Dict, Union, TypeVar, Mapping, Callable, Optional
from abc import ABC, abstractmethod
from queue import PriorityQueue
from collections import defaultdict
import heapq
import pickle

np.random.seed(127)

from base import plot_points, Point, compute_pareto_frontier_naive, compute_pareto_frontier_efficient
from util import hierarchy_pos

class GraphPart:
    '''
    A random part of a graph with no particular structure 
    '''
    def __init__(self, name, graph, opt_confs=None):
        self.graph = graph             
        self.name = name
        self._frontier = None 
        self.nx_graph = self._to_nx_digraph()

    def _to_nx_digraph(self):
        g = nx.DiGraph()
        for node, deps in self.graph.items():
            for dep in deps:
                g.add_edge(dep, node)
        
        return g
    
    def get_all_nodes(self):
        '''
        Returns all nodes of this graph recursively 
        '''
        all_nodes = set()
        for node in self.nx_graph.nodes():
            all_nodes.add(node)
            if node.nx_graph.nodes() > 1:
                all_nodes.update(node.get_all_nodes())
        
        return all_nodes 
    
    def draw(self, sz=1000, color=None, layout='tree'):
        base_size = sz / 2
        g = self.nx_graph
        node_size = [len(str(node)) * base_size for node in g.nodes()]
        if layout == 'tree':
            pos = hierarchy_pos(g.reverse())
        else:
            pos = nx.nx_pydot.graphviz_layout(g)
            
        nx.draw(g, pos, with_labels=True, node_size=node_size, node_color=color) 
        
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
        
    def __repr__(self):
        return self.name
        
    def dependencies(self, x):
        return self.graph.get(x, ())
    
    def dependants(self, x):
        return list(self.nx_graph.successors(x))
        
    def _evaluate(self, raw_config) -> Point:
        config = dict()
        for (stage, point) in raw_config:
            config[stage] = point
        
        def incoming_latency(v):
            nonlocal self, config
            max_child = max((incoming_latency(c) for c in self.dependencies(v)), default=0)
            return config[v].latency + max_child
        
        latency = incoming_latency(self.root)
        cost = sum(p.cost for p in config.values())
        return Point(latency=latency, cost=cost)        
    
    def compute_all_configurations(self, verbose=False, old=False, stop_after=None):        
        pre_product = []
        product_size = 1
        topsort_order = list(nx.topological_sort(self.nx_graph))
        position_of = dict()
        for idx, node in enumerate(topsort_order):
            pre_product.append([(node, point) for point in node.frontier])
            product_size *= len(node.frontier)
            position_of[node] = idx
        
        def mass_evaluate(raw_config):
            nonlocal position_of
            latency = dict()
            cost = 0
            for node in topsort_order:
                node_point = raw_config[position_of[node]][1]
                node_latency = node_point.latency
                if node in self.graph:
                    node_latency += max((latency[child] for child in self.graph[node]), default=0)
                    
                latency[node] = node_latency
                cost += node_point.cost
            
            return Point(latency=latency[topsort_order[-1]], cost=cost)
        
        all_confs = itertools.product(*pre_product)
        if verbose:
            ans = []
            for conf in tqdm(all_confs, total=product_size):
                ans.append(mass_evaluate(conf))
            return ans
        else:
            if old:
                return list(map(self._evaluate, all_confs))
            else:
                if stop_after is not None:
                    ans = []
                    for idx, conf in enumerate(all_confs):
                        ans.append(mass_evaluate(conf))
                        if idx >= stop_after:
                            break
                    
                    return ans 
                else:
                    return list(map(mass_evaluate, all_confs))
        
    def compute_frontier(self, max_pts=None):
        return compute_pareto_frontier_efficient(self.compute_all_configurations())
        
    def plot_frontier(self, plt):
        all_points = self.compute_all_configurations()
        plot_points(plt, all_points, label='All configurations')
        plot_points(plt, self.frontier, label='Pareto frontier')
        
    @property
    def root(self):
        if not hasattr(self, '_root'):
            g = self.nx_graph
            self._root = [x for x in g.nodes() if g.out_degree(x) == 0][0]
            
        return self._root    
    
    def set_frontier(self, new_front):
        self._frontier = new_front
    
    @property
    def frontier(self):
        if self._frontier is None:
            self._frontier = self.compute_frontier()
        
        return self._frontier

class SingleNode(GraphPart):
    '''
    Idea: replace the Stage class 
    '''
    def __init__(self, name, frontier):
        super().__init__(name, dict())
        self._root = self 
        self._frontier = frontier 
        self.nx_graph.add_node(self)
        
    def compute_frontier(self, max_pts=None):
        return self._frontier
    
    @classmethod 
    def random(cls, k_points=5):
        TYPES = (
            (2, 30),
            (7, 70),
            (10, 20),
            (5, 15),
            (15, 50),
            (12, 84),
            (48, 10)
        )
        
        def _gen_pareto_points(n, latency, coeff):
            # cost * latency ~= coeff 
            latencies = np.random.uniform(latency / 2, 2 * latency, n)
            costs = coeff / latencies
            confs = [Point(latency=lat, cost=c) for (lat, c) in zip(latencies, costs)]
            return confs
        
        stage_type = np.random.choice(np.arange(0, len(TYPES)))
        latency, coeff = TYPES[stage_type]
        frontier = _gen_pareto_points(k_points, latency, coeff)
        name = np.random.choice(list(string.ascii_uppercase)) \
               + np.random.choice(list(string.ascii_uppercase)) \
               + np.random.choice(list(string.digits[1:]))
        
        return cls(name, frontier)

class List(GraphPart):
    def compute_frontier(self, max_pts=None):
        def merge(node1, node2):
            name = f'{node1.name}+{node2.name}'
            all_points = List.from_node_list([node1, node2]).compute_all_configurations()
            nonlocal max_pts
            return SingleNode(name, compute_pareto_frontier_efficient(all_points))
        
        def divide_and_conquer(nodes):
            if len(nodes) == 1:
                return nodes[0]
            if len(nodes) == 2:
                return merge(nodes[0], nodes[1])
                
            m = len(nodes) // 2
            left = divide_and_conquer(nodes[:m])
            right = divide_and_conquer(nodes[m:])
            return merge(left, right)

        return divide_and_conquer(self.initial_stages).compute_frontier(max_pts=max_pts)
        
    @classmethod 
    def from_node_list(cls, stages):
        graph = {}
        for src, dest in zip(stages[:-1], stages[1:]):
            graph[src] = [dest]
        
        name = '+'.join([s.name for s in stages])
        obj = cls(name, graph)
        obj.initial_stages = stages     
        return obj
    
    @classmethod
    def random(cls, n_nodes=5, k_points=5):
        nodes = [SingleNode.random(k_points) for _ in range(n_nodes)]
        return cls.from_node_list(nodes)

class Join(GraphPart):
    def compute_frontier(self, max_pts=None):
        def merge(node1, node2):
            nonlocal max_pts
            name = f'{node1.name}+{node2.name}'
            all_points = [Point(latency=max(p1.latency, p2.latency), cost=p1.cost + p2.cost)
                          for p1 in node1.compute_frontier(max_pts)
                          for p2 in node2.compute_frontier(max_pts)]
            return SingleNode(name, compute_pareto_frontier_efficient(all_points))
        
        def divide_and_conquer(nodes):
            if len(nodes) == 1:
                return nodes[0]
            if len(nodes) == 2:
                return merge(nodes[0], nodes[1])
                
            m = len(nodes) // 2
            left = divide_and_conquer(nodes[:m])
            right = divide_and_conquer(nodes[m:])
            return merge(left, right)
        
        compressed_children = divide_and_conquer(self.children)
        return List.from_node_list([self.parent, compressed_children]).compute_frontier(max_pts=max_pts)

    @property
    def parent(self):
        return list(self.graph.keys())[0]
    
    @property
    def children(self):
        return self.graph[self.parent]
    
    @classmethod 
    def from_parent_and_children(cls, parent, children):
        graph = {parent: []}
        for child in children:
            graph[parent].append(child)
            
        name = f'{parent.name}|({",".join([c.name for c in children])})'
        obj = cls(name, graph)        
        return obj
    
    @classmethod
    def random(cls, n_children=3, k_points=5):
        parent = SingleNode.random(k_points)
        children = [SingleNode.random(k_points) for _ in range(n_children)]
        return cls.from_parent_and_children(parent, children)

Graph = TypeVar('Graph')

class Graph(GraphPart):
    '''
    Only useful for working with entire graphs
    '''        
    def all_chidren(self, node):
        ans = set()
        def dfs(x):
            nonlocal self, ans 
            ans.add(x)
            for c in self.dependencies(x):
                dfs(c)
        
        dfs(node)
        return ans 
    
    def remove_edge(self, src, dest):
        if self.nx_graph.has_edge(dest, src):
            self.graph[src] = [x for x in self.dependencies(src) if x != dest]
            self.nx_graph.remove_edge(dest, src)
    
    def _remove_direct_dependencies(ans: Graph):
        for node in ans.detect_unresolved():
            parents = ans.dependants(node)
            for idx1, p1 in enumerate(parents):
                for p2 in parents[idx1 + 1:]:
                    if p1 in ans.all_chidren(p2):
                        ans.remove_edge(p2, node)
                    elif p2 in ans.all_chidren(p1):
                        ans.remove_edge(p1, node)
        
    def _remove_indirect_dependencies(ans: Graph):
        for node in ans.detect_unresolved():
            max_latency_node = ans.max_latency_at(node)
            for parent in ans.dependants(node):
                other_incoming_latency = 0
                for child in ans.dependencies(parent):
                    if child == node:
                        continue 
                     
                    other_incoming_latency = max(ans.min_latency_at(child), other_incoming_latency)
                
                if other_incoming_latency >= max_latency_node:
                    ans.remove_edge(parent, node)
                    if len(ans.dependants(node)) == 1:
                        break
    
    def without_unnecessary_dependencies(self, indirect=True) -> Graph:
        ans = Graph(self.name + '_nud', {k:list(v) for k, v in self.graph.items()})
        Graph._remove_direct_dependencies(ans)
        if indirect:
            Graph._remove_indirect_dependencies(ans)
        return ans 
                        
    def detect_unresolved(self):
        return set(x for x in self.nx_graph.nodes() if len(self.dependants(x)) > 1)
        
    def with_compressed_lists(self) -> Graph:
        new_g = {k:list(v) for k, v in self.graph.items()}
        handled = set()
        
        def dependencies(node):
            nonlocal new_g
            return new_g.get(node, ())
        
        def dependants(node):
            nonlocal new_g
            return [x for x in new_g.keys() if node in new_g[x]]
        
        def replace_list(lower, upper):
            nonlocal new_g, handled
            list_parents = dependants(upper)
            list_children = dependencies(lower)
            lst = []
            x = upper
            while x != lower:
                lst.append(x)
                x = dependencies(x)[0]

            lst.append(lower)

            for v in lst:
                handled.add(v)
                if v in new_g:
                    del new_g[v]

            lst_node = List.from_node_list(lst)
            for list_parent in list_parents:
                new_g[list_parent] = [node for node in dependencies(list_parent) if node != upper] 
                new_g[list_parent].append(lst_node)

            new_g[lst_node] = list_children
        
        for node in self.nx_graph.nodes():
            if node in handled:
                continue 
            
            if len(dependencies(node)) == 1:
                # definitely a list element 
                # now find its boundaries 
                upper = node
                x = node 
                while len(dependants(x)) == 1 and len(dependencies(x)) == 1:
                    upper = x
                    x = dependants(x)[0]
                
                lower = node
                x = node 
                while len(dependencies(x)) == 1:
                    lower = x
                    x = dependencies(x)[0]
                    if len(dependants(x)) == 1:
                        lower = x 
                    else:
                        break
    
                # now upper contains the upper part of the list 
                # lower contains the lower part of the list 
                replace_list(lower, upper)
                
        return Graph(self.name + '_cl', new_g)
    
    def subtree_rooted_at(self, node):
        subgraph = dict()
        for x in self.all_chidren(node):
            subgraph[x] = self.dependencies(x)
            if x != node and len(self.dependants(x)) >= 2:
                return None 
        
        return Graph(f'({node.name})^T', subgraph)
    
    def with_compressed_subtrees(self):
        new_graph = dict()
        cache = dict()
        def dfs(x):
            nonlocal self, new_graph
            if x in cache:
                return cache[x]
            
            if len(self.dependencies(x)) == 0:
                return x
            
            sub = self.subtree_rooted_at(x)
            if sub is None:
                new_graph[x] = []
                for c in self.dependencies(x):
                    new_graph[x].append(dfs(c))
                    
                return x
            else:
                cache[x] = sub
                return sub
        
        dfs(self.root)
        return Graph(self.name + '_cst', new_graph)
    
    def _compute_frontier_tree(self):
        def make_join(x):
            nonlocal self 
            # avoid lists 
            # assert len(self.dependencies(x)) >= 2
            children = []
            for child in self.dependencies(x):
                if len(self.dependencies(child)) > 0: 
                    children.append(make_join(child))
                else:
                    children.append(child)
            
            return Join.from_parent_and_children(parent=x, children=children)
        
        return make_join(self.root).frontier

    def min_latency_at(self, node):
        max_child = max((self.min_latency_at(c) for c in self.dependencies(node)), default=0)
        return min(p.latency for p in node.frontier) + max_child
    
    def max_latency_at(self, node):
        max_child = max((self.max_latency_at(c) for c in self.dependencies(node)), default=0)
        return max(p.latency for p in node.frontier) + max_child
    
    def compute_frontier(self):
        assert len(self.detect_unresolved()) == 0
        return self._compute_frontier_tree()
    
    def approximate_frontier(self, compute_threshold_size: Optional[Callable[[int], int]] = None):
        pre_product = []
        product_size = 1
        frontier_heap = []
        new_frontiers = dict()
        for node in self.nx_graph.nodes():
            front_size = len(node.frontier)
            frontier_heap.append((-1 * front_size, node.name, node))
            product_size *= front_size
            new_frontiers[node] = list(node.frontier)
        
        heapq.heapify(frontier_heap)      
        target_size = 100_000
        if compute_threshold_size is not None:  
            target_size = compute_threshold_size(len(self.nx_graph.nodes))
            
        while product_size > target_size:
            top_size, _, top_node = heapq.heappop(frontier_heap)
            top_size *= -1 
            # randomly sample 1/2 of the frontier 
            new_frontiers[top_node] = np.random.choice(new_frontiers[top_node], top_size // 2, replace=False)
            product_size = (product_size // top_size) * (top_size // 2)
            heapq.heappush(frontier_heap, (-1 * top_size // 2, top_node.name, top_node))
             
        topsort_order = list(nx.topological_sort(self.nx_graph))
        position_of = dict()
        for idx, node in enumerate(topsort_order):
            pre_product.append([(node, point) for point in new_frontiers[node]])
            product_size *= len(new_frontiers[node])
            position_of[node] = idx
        
        def mass_evaluate(raw_config):
            latency = dict()
            cost = 0
            for node in topsort_order:
                node_point = raw_config[position_of[node]][1]
                node_latency = node_point.latency
                if node in self.graph:
                    node_latency += max((latency[child] for child in self.graph[node]), default=0)
                    
                latency[node] = node_latency
                cost += node_point.cost
            
            return Point(latency=latency[topsort_order[-1]], cost=cost)
        
        all_confs = itertools.product(*pre_product)
        return compute_pareto_frontier_efficient(list(map(mass_evaluate, all_confs)))

def graph_pareto_front_naive(g: Graph):
    return GraphPart(g.name, g.graph).compute_frontier()

def dag_frontier_best_effort(g: Graph, compute_threshold_size: Optional[Callable[[int], int]] = None):
    new_g = g.without_unnecessary_dependencies(indirect=True)
    if len(new_g.detect_unresolved()) == 0:
        return new_g.compute_frontier()
    else:
        return new_g.with_compressed_lists() \
                    .with_compressed_subtrees() \
                    .approximate_frontier(compute_threshold_size=compute_threshold_size)

def compute_utopia(g: Graph):
    latency = g.min_latency_at(g.root)
    cost = sum(min(p.cost for p in x.frontier) for x in g.nx_graph.nodes())
    return np.array([latency, cost])

def compute_nadir(g: Graph):
    latency = g.max_latency_at(g.root)
    cost = sum(max(p.cost for p in x.frontier) for x in g.nx_graph.nodes())
    return np.array([latency, cost])

def plot_utopia_nadir(ax, g):
    nadir = compute_nadir(g)
    utopia = compute_utopia(g)
    x = [nadir[0], utopia[0]]
    y = [nadir[1], utopia[1]]
    ax.scatter(x, y, label='Reference points')