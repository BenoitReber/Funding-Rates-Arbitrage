import time
from collections import deque
from graphviz import Digraph

from core.data.models.time_series import *

# Module for managing time series dependencies and updates
# This module provides classes for building and managing dependency graphs
# between time series, ensuring they are updated in the correct order

class DependencyGraph:
    """
    A directed graph structure for managing dependencies between time series.
    
    This class builds and maintains a directed acyclic graph (DAG) of time series
    dependencies, where each node is a TimeSerie object and edges represent dependencies.
    It provides methods for adding series, topological sorting, and visualization.
    
    Attributes
    ----------
    graph : dict
        Dictionary mapping each series to a set of its dependencies.
    nodes : set
        Set of all series (nodes) in the graph.
    """
    def __init__(self):
        """
        Initialize an empty dependency graph.
        
        Creates an empty graph structure with no nodes or edges.
        """
        self.graph = {}  # Dictionary mapping each series to a set of its dependencies
        self.nodes = set()  # Set of all series (nodes) in the graph

    def add_series(self, series):
        """
        Recursively register a series and all its dependencies in the graph.
        
        This method adds a time series to the graph along with all its dependencies.
        If the series is already in the graph, it does nothing to avoid duplicates.
        For each dependency of the series, it recursively adds that dependency as well.
        
        Parameters
        ----------
        series : TimeSerie
            The time series to add to the dependency graph.
        
        Notes
        -----
        - Uses recursion to add all dependencies in the dependency tree
        - Handles circular dependencies implicitly (they will be detected during topological sort)
        """
        if series not in self.nodes:
            # Add the series to the set of nodes if it's not already there
            self.nodes.add(series)
            
            # Get the dependencies of the series (empty list if none)
            deps = series.dependencies if series.dependencies else []
            
            # Add the dependencies to the graph entry for this series
            self.graph[series] = set(deps)
            
            # Recursively add each dependency to the graph
            for dep in deps:
                self.add_series(dep)  # Add dependencies recursively
        
    def topological_sort(self):
        """
        Perform topological sorting of the dependency graph using Kahn's algorithm.
        
        This method sorts the nodes (time series) such that for every directed edge
        from node A to node B, node A appears before node B in the sorted order.
        This ensures that all dependencies of a series are updated before the series itself.
        
        Returns
        -------
        list
            A list of TimeSerie objects in topologically sorted order.
            The order is reversed so that dependencies come first.
        
        Raises
        ------
        ValueError
            If a cycle is detected in the dependency graph, making topological sort impossible.
        
        Notes
        -----
        - Uses Kahn's algorithm for topological sorting
        - Detects cycles in the dependency graph
        - Returns the reverse of the standard topological sort to ensure dependencies are processed first
        """
        # Initialize in-degree (number of dependencies) for each node
        in_degree = {node: 0 for node in self.nodes}
        for node in self.graph:
            for dep in self.graph[node]:
                in_degree[dep] += 1  # Count how many series depend on each node

        # Start with nodes that have no dependencies (in-degree = 0)
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        sorted_order = []
        
        # Process nodes in order of dependency resolution
        while queue:
            node = queue.popleft()  # Get next node with no unprocessed dependencies
            sorted_order.append(node)  # Add to result
            
            # Update in-degree for nodes that depend on this one
            for dependent in self.graph[node]:
                in_degree[dependent] -= 1  # Decrement in-degree
                if in_degree[dependent] == 0:  # If all dependencies processed
                    queue.append(dependent)  # Add to queue
                    
        # Check for cycles - if not all nodes are in sorted_order, there's a cycle
        if len(sorted_order) != len(self.nodes):
            raise ValueError("Cycle detected in dependency graph")

        # Reverse to get dependencies first, then dependents
        sorted_order.reverse()    
        
        return sorted_order

    def _get_dependents(self, node):
        """
        Find all series that directly depend on the given node.
        
        Parameters
        ----------
        node : TimeSerie
            The time series node to find dependents for.
            
        Returns
        -------
        list
            A list of TimeSerie objects that directly depend on the given node.
        
        Notes
        -----
        - This is a helper method used internally
        - The method name starts with underscore to indicate it's not part of the public API
        """
        # List comprehension to find all nodes that have the given node as a dependency
        return [n for n in self.graph if node in self.graph[n]]
    
    def visualize(self):
        """
        Visualize the dependency graph using Graphviz.
        
        Creates a visual representation of the dependency graph where each node
        is a time series and edges represent dependencies. The graph is rendered
        to a file and displayed.
        
        Returns
        -------
        Digraph
            The Graphviz Digraph object representing the dependency graph.
            
        Notes
        -----
        - Requires the graphviz package to be installed
        - Saves the graph to 'dependencies.gv' and opens it automatically
        - TODO: Add parameters to control file output and display options
        """
        # Create a new directed graph
        dot = Digraph()
        
        # Add nodes to the graph with metadata as labels
        for node in self.nodes:
            dot.node(str(id(node)), str(node.get_metadata()))
            
        # Add edges from dependencies to dependents
        for node, deps in self.graph.items():
            for dep in deps:
                dot.edge(str(id(dep)), str(id(node)))
                
        # Render the graph to a file and display it
        dot.render('dependencies.gv', view=True)
        return dot
    
class SeriesManager:
    """
    Manager for a collection of time series with dependencies.
    
    This class manages a collection of time series objects, handling their dependencies
    and ensuring they are updated in the correct order. It maintains a dependency graph
    and provides methods for registering series and updating them according to their
    dependencies and update intervals.
    
    Attributes
    ----------
    graph : DependencyGraph
        The dependency graph for all managed time series.
    series : list
        List of all registered time series.
    _sorted_cache : list or None
        Cached topologically sorted list of series for efficient updates.
    update_intervals : dict
        Dictionary mapping series to their update intervals in milliseconds.
    timestamp : int
        Current timestamp in milliseconds.
    _smallest_duration : int
        The smallest update interval among all registered series.
    """
    def __init__(self, timestamp=None):
        """
        Initialize a new SeriesManager.
        
        Parameters
        ----------
        timestamp : int, optional
            Initial timestamp in milliseconds. If None, defaults to 0.
            
        Notes
        -----
        - Creates an empty dependency graph
        - Initializes with no registered series
        - Sets the smallest duration to -1 (invalid) until series are registered
        """
        self.graph = DependencyGraph()  # Create a new dependency graph
        self.series = []  # List to store all registered series
        self._sorted_cache = None  # Cache for topologically sorted series
        self.update_intervals = {}  # Dictionary mapping series to update intervals
        self.timestamp = timestamp if timestamp else 0  # Current timestamp
        self._smallest_duration = -1  # Smallest update interval (initialized as invalid)
        
    def register(self, series):
        """
        Register a new time series with the manager.
        
        This method adds a time series to the manager, updates the dependency graph,
        and recalculates the topological sort and smallest duration.
        
        Parameters
        ----------
        series : TimeSerie
            The time series to register with the manager.
            
        Notes
        -----
        - Adds the series to the internal list and dependency graph
        - Updates the update intervals dictionary with the series duration
        - Recalculates the smallest duration among all series
        - Rebuilds the topologically sorted cache
        """
        # Add the series to the list of managed series
        self.series.append(series)
        
        # Add the series to the dependency graph (including its dependencies)
        self.graph.add_series(series)
        
        # Invalidate the sorted cache (will be rebuilt below)
        self._sorted_cache = None
        
        # Store the update interval for this series
        self.update_intervals[series] = series.duration
        
        # Update the smallest duration among all series
        self._smallest_duration = min(list(self.update_intervals.values()))
        
        # Rebuild the topologically sorted cache
        self._sorted_cache = self.graph.topological_sort()
        
    def update_all_old(self):
        """
        Update all series in correct order based on real-time and intervals.
        
        This method updates all series in topological order (dependencies first),
        but only if they need to be updated based on their update intervals and
        the current time. It continues updating until the internal timestamp
        catches up with the current real time.
        
        Notes
        -----
        - This is an older implementation kept for reference
        - Updates series only when their interval boundary is crossed
        - Continues updating until the timestamp catches up with real time
        - Increments the timestamp by the smallest duration after each update cycle
        """
        # Ensure the sorted cache is initialized
        if not self._sorted_cache:
            self._sorted_cache = self.graph.topological_sort()

        # Continue updating until we catch up with real time
        # Compares the current interval with the timestamp interval
        while int((time.time() * 1000) / self._smallest_duration) > int(self.timestamp / self._smallest_duration):

            # Update each series in topological order (dependencies first)
            for series in self._sorted_cache:
                # Check if the series needs to be updated based on intervals
                tmp_duration = self.update_intervals[series]
                current_interval = int(self.timestamp / tmp_duration)
                last_interval = int((series.timestamps[-1]) / tmp_duration)
                
                # Only update if we've crossed an interval boundary
                if last_interval < current_interval:
                    series.update()
            
            # Increment the timestamp by the smallest duration
            self.timestamp += self._smallest_duration
    
    def update_all_old_bis(self):
        """
        Update all BuiltSerie instances when their update interval is reached.
        
        This method updates BuiltSerie instances in topological order, but only
        when the current timestamp is exactly divisible by their update interval.
        After updating, it increments the timestamp by the smallest duration.
        
        Notes
        -----
        - This is an older implementation kept for reference
        - Only updates BuiltSerie instances (not ThirdPartySerie)
        - Updates only when timestamp is exactly divisible by the update interval
        - Increments the timestamp by the smallest duration after all updates
        """
        # Ensure the sorted cache is initialized
        if not self._sorted_cache:
            self._sorted_cache = self.graph.topological_sort()

        # Update each series in topological order (dependencies first)
        for series in self._sorted_cache:
            # Check if the series is a BuiltSerie and needs to be updated
            # Only update when the timestamp is exactly divisible by the update interval
            if type(series) == BuiltSerie and self.timestamp % self.update_intervals[series] == 0:
                series.update()
        
        # Increment the timestamp by the smallest duration
        self.timestamp += self._smallest_duration

    def update_all(self, verbose=False):
        """
        Update all BuiltSerie instances in dependency order.
        
        This method updates all BuiltSerie instances in topological order,
        ensuring that dependencies are updated before the series that depend on them.
        Unlike the older versions, this method updates all BuiltSerie instances
        regardless of their update intervals or the current timestamp.
        
        Notes
        -----
        - This is the current implementation used for updates
        - Only updates BuiltSerie instances (not ThirdPartySerie)
        - Updates all BuiltSerie instances unconditionally
        - Does not increment the timestamp after updates
        """
        # Ensure the sorted cache is initialized
        if not self._sorted_cache:
            self._sorted_cache = self.graph.topological_sort()

        # Update each BuiltSerie in topological order (dependencies first)
        for series in self._sorted_cache:
            # Only update BuiltSerie instances (not ThirdPartySerie)
            if type(series) == BuiltSerie:
                series.update(verbose=verbose)