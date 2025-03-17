import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_coordinates(file_path):
    """
    Read coordinates from CSV file
    """
    df = pd.read_csv(file_path)
    return df[['x', 'y']].values

def calculate_distance(point1, point2):
    """ 
    Calculate Euclidean distance between two points
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_travel_time(distance, passengers):
    """
    Calculate travel time based on distance and number of passengers
    Using the formula: v = v₀/(1 + αp)
    where:
    - v₀ = base speed (1 unit per hour)
    - p = number of passengers
    - α = 0.01 (speed reduction factor)
    """
    base_speed = 1.0  # v₀
    alpha = 0.01     # α
    speed = base_speed / (1 + alpha * passengers)
    return distance / speed

def is_valid_move(from_point, to_point):
    """
    Check if a move is valid based on wind constraints
    Movement from South to North is not allowed
    """
    # Calculate the direction vector
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    
    # If moving directly South to North (dy > 0), it's invalid
    if dx == 0 and dy > 0:
        return False
    
    return True

def find_route(coordinates):
    """
    Find the optimal route using Dynamic Programming with Branch & Bound.
    Uses memoization for overlapping subproblems and pruning for efficiency.
    
    The algorithm works by:
    1. Using Dynamic Programming to cache distances and times
    2. Using Branch & Bound to efficiently search through possible routes
    3. Using pruning to skip paths that can't be optimal
    4. Trying all possible starting points to find the globally optimal route
    
    Args:
        coordinates: Array of (x,y) coordinates for each site
    Returns:
        best_route: List of site indices in optimal order
        best_time: Total travel time for the optimal route
    """
    n = len(coordinates)
    best_route = None
    best_time = float('inf')
    
    # Cache dictionaries to store pre-calculated values
    # This is the Dynamic Programming part - we avoid recalculating the same values
    distance_cache = {}  # Stores distances between pairs of points
    time_cache = {}     # Stores travel times between pairs of points with given passenger counts
    
    def get_distance(i, j):
        """
        Get distance between points i and j, using cache to avoid recalculation.
        This is part of the Dynamic Programming optimization.
        """
        if (i, j) not in distance_cache:
            distance_cache[(i, j)] = calculate_distance(coordinates[i], coordinates[j])
        return distance_cache[(i, j)]
    
    def get_time(i, j, passengers):
        """
        Get travel time between points i and j with given passengers, using cache.
        This is part of the Dynamic Programming optimization.
        """
        if (i, j, passengers) not in time_cache:
            distance = get_distance(i, j)
            time_cache[(i, j, passengers)] = calculate_travel_time(distance, passengers)
        return time_cache[(i, j, passengers)]
    
    def lower_bound(current, remaining, current_time, passengers):
        """
        Calculate a lower bound on the remaining time using the minimum possible times
        for remaining moves. This is part of the Branch & Bound optimization.
        
        The lower bound helps us prune branches that can't possibly lead to a better solution.
        It does this by:
        1. Looking at all remaining valid moves
        2. Finding the minimum possible time for each move
        3. Summing these minimum times to get a lower bound on remaining time
        
        If current_time + lower_bound >= best_time, we can skip this branch because
        even the best possible completion of this path can't be better than our current best.
        """
        if not remaining:
            return 0
        
        # Find minimum possible time for each remaining move
        min_times = []
        for next_point in remaining:
            if is_valid_move(coordinates[current], coordinates[next_point]):
                min_times.append(get_time(current, next_point, passengers))
        
        if not min_times:
            return float('inf')
        
        # Add minimum times for remaining moves
        return sum(min_times)
    
    def branch_and_bound(current, remaining, current_route, current_time, passengers):
        """
        This function finds the best route by systematically trying all possible paths
        while using clever shortcuts (pruning) to skip paths that can't be optimal.
        """
        nonlocal best_route, best_time  # Allows us to update the best route found so far
        
        # STEP 1: Check if we've found a complete route
        if not remaining:  # If no points left to visit
            if current_time < best_time:  # If this route is better than our best so far
                best_time = current_time
                best_route = current_route.copy()
            return
        
        # STEP 2: Calculate a lower bound for remaining moves
        lb = lower_bound(current, remaining, current_time, passengers)
        
        # STEP 3: Skip this branch if it can't be better than our best
        if current_time + lb >= best_time:
            return
        
        # STEP 4: Try all possible next moves
        for next_point in remaining:
            # Check if this move is valid (respects wind constraints)
            if is_valid_move(coordinates[current], coordinates[next_point]):
                # Calculate time for this move
                time = get_time(current, next_point, passengers)
                
                # Skip if this path can't be better than our best
                if current_time + time >= best_time:
                    continue
                
                # STEP 5: Try this path
                # First, update our state
                remaining.remove(next_point)  # Remove from unvisited points
                current_route.append(next_point)  # Add to our current route
                
                # Recursively try this path (DFS)
                branch_and_bound(
                    next_point,  # New current point
                    remaining,   # Remaining unvisited points
                    current_route,  # Current route so far
                    current_time + time,  # Updated total time
                    passengers + 10  # Add 10 new passengers
                )
                
                # STEP 6: Backtrack
                # Restore our state for the next iteration
                remaining.add(next_point)  # Put point back in unvisited set
                current_route.pop()  # Remove from current route
    
    # Try starting from each point to find the globally optimal route
    for start_point in range(n):
        remaining = set(range(n))
        remaining.remove(start_point)
        branch_and_bound(start_point, remaining, [start_point], 0, 10)  # Start with 10 passengers
    
    return best_route, best_time

def visualize_route(coordinates, route):
    """
    Visualize the route and sites
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all sites
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, label='Sites')
    
    # Plot the route
    route_coords = coordinates[route]
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'r-', label='Route')
    
    # Add site numbers
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(f'Site {i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title('Navigation Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Read coordinates from CSV file
    coordinates = read_coordinates('1_nuanced_navigation.csv')
    
    # Find the optimal route
    route, total_time = find_route(coordinates)
    
    if route:
        # Print results
        print("\nOptimal Route:")
        print(" -> ".join([f"Site {i}" for i in route]))
        print(f"\nTotal Travel Time: {total_time:.2f} hours")
        
        # Visualize the route
        visualize_route(coordinates, route)
    else:
        print("Could not find a valid route!")

if __name__ == "__main__":
    main() 