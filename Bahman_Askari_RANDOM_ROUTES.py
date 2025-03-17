"""
Random Routes - Pathfinding with Geometric Constraints
----------------------------------------------------
This program finds the maximum-value path in a 2D point set where:
- Each point has a value between 0 and 1
- Path must be continuous and non-crossing
- Each point can be visited at most once
- Angles between consecutive segments must be > 90 degrees
- Path can start and end at any different points

Input: CSV file containing point coordinates (x, y) and values
Output: Maximum value path and visualization of the solution
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Set


def read_points_from_csv(filename: str) -> pd.DataFrame:
    """
    Read points from CSV file.
    Each point is defined by its coordinates (x, y) and a value.

    Args:
        filename: Path to CSV file containing point data with columns:
                 x, y, value

    Returns:
        DataFrame with columns x, y, value
    """
    try:
        # Read CSV file into pandas DataFrame
        df = pd.read_csv(filename)

        # Convert columns to float and remove any invalid rows
        df = df.astype(float).dropna()

        print(f"Successfully loaded {len(df)} points from {filename}")
        return df

    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        raise


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Calculate the angle between three points in degrees.
    Uses vector operations to compute the angle between line segments.
    Returns the angle between the two segments (-180 to 180 degrees).
    Positive angle means turning left, negative means turning right.

    Args:
        p1, p2, p3: Points as (x, y) tuples representing three consecutive points

    Returns:
        float: Angle in degrees between the two line segments
    """
    # Convert points to numpy arrays for vector operations
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # Vector from p1 to p2
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])  # Vector from p2 to p3

    # Calculate angle using dot product formula: cos(θ) = (v1·v2)/(|v1||v2|)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)  # Length of first vector
    norm2 = np.linalg.norm(v2)  # Length of second vector

    # Handle edge case where vectors are zero length
    if norm1 == 0 or norm2 == 0:
        return 0

    # Calculate angle in degrees
    cos_angle = dot_product / (norm1 * norm2)
    # Clamp cos_angle to [-1, 1] to avoid numerical errors in arccos
    cos_angle = max(min(cos_angle, 1), -1)
    angle = np.degrees(np.arccos(cos_angle))

    # Determine if the angle is positive (left turn) or negative (right turn)
    # using the cross product of v1 and v2
    cross_product = np.cross(v1, v2)
    if cross_product < 0:
        angle = -angle

    # For debugging
    print(f"Angle between segments: {angle:.2f}°")

    return angle


def segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                       p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    Check if two line segments intersect using CCW (Counter Clockwise) method.

    The CCW method determines if a point C is counter-clockwise from point B
    when looking from point A. This is used to determine if line segments intersect.

    Example:
        A(0,0) ---- B(2,0)
         |           |
         |           |
        C(0,2) ---- D(2,2)

        ccw(A,B,C) returns True because C is counter-clockwise from B when looking from A
        ccw(A,B,D) returns False because D is clockwise from B when looking from A

    Args:
        p1, p2: Endpoints of first line segment
        p3, p4: Endpoints of second line segment

    Returns:
        bool: True if segments intersect, False otherwise
    """

    def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """
        Determine if point C is counter-clockwise from point B when looking from point A.

        The function uses the cross product of vectors AB and AC:
        - If result > 0: C is counter-clockwise from B
        - If result < 0: C is clockwise from B
        - If result = 0: C is collinear with A and B

        Args:
            A: Reference point (origin)
            B: First point
            C: Second point

        Returns:
            bool: True if C is counter-clockwise from B, False otherwise
        """
        # Calculate cross product of vectors AB and AC
        # (C[1]-A[1]) * (B[0]-A[0]) is the y-component of AB × AC
        # (B[1]-A[1]) * (C[0]-A[0]) is the x-component of AB × AC
        # The difference gives us the z-component of the cross product
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Two line segments intersect if and only if:
    # 1. One endpoint of each segment is on opposite sides of the other segment
    # 2. The endpoints of one segment are on opposite sides of the other segment
    #
    # This is checked using the CCW function:
    # - ccw(p1,p3,p4) != ccw(p2,p3,p4): p1 and p2 are on opposite sides of p3-p4
    # - ccw(p1,p2,p3) != ccw(p1,p2,p4): p3 and p4 are on opposite sides of p1-p2
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def construct_graph(df: pd.DataFrame) -> List[List[Tuple[int, float]]]:
    """
    Construct a graph where:
    - Nodes are points with their values as weights
    - Edges are created based on Euclidean distance and angle constraints
    - Graph is represented as an adjacency list

    Args:
        df: DataFrame with columns x, y, value containing all points

    Returns:
        List of lists where each inner list contains tuples of (neighbor_index, edge_weight)
        Edge weight is the sum of the values of the connected points
    """
    n = len(df)
    graph = [[] for _ in range(n)]  # Adjacency list representation

    # For each point, find valid connections to other points
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if points are the same
            if i == j:
                continue

            # Calculate edge weight (sum of point values)
            edge_weight = df.iloc[i]['value'] + df.iloc[j]['value']

            # Add bidirectional edge
            graph[i].append((j, edge_weight))
            graph[j].append((i, edge_weight))

    return graph


def find_max_value_path(df: pd.DataFrame) -> Tuple[List[int], float]:
    """
    Find the path with maximum cumulative value that satisfies angle and crossing constraints.

    Args:
        df: DataFrame with point data

    Returns:
        Tuple containing:
        - List of point indices in the optimal path
        - Maximum cumulative value
    """
    # Step 1: Construct the graph
    graph = construct_graph(df)

    # Step 2: Find the point with maximum value to start from
    start_points = df.nlargest(10, 'value').index.tolist()  # Try top 10 points
    best_path = []
    best_value = float('-inf')

    # Step 3: Try each potential start point
    for start_idx in start_points:
        # Step 4: Initialize data structures for Dijkstra's algorithm
        distances = [float('-inf')] * len(df)  # Use negative infinity for maximum path
        distances[start_idx] = df.iloc[start_idx]['value']  # Start with the point's value
        previous = [-1] * len(df)  # For path reconstruction
        visited = set()

        # Step 5: Main Dijkstra's algorithm loop
        while len(visited) < len(df):
            # Find unvisited node with maximum distance
            current = -1
            max_dist = float('-inf')
            for i in range(len(df)):
                if i not in visited and distances[i] > max_dist:
                    max_dist = distances[i]
                    current = i

            if current == -1:
                break

            visited.add(current)

            # Step 6: Update distances to all neighbors
            for neighbor, weight in graph[current]:
                # Skip if neighbor is already visited
                if neighbor in visited:
                    continue

                # Step 7: Check angle constraint
                # Only check if we have a previous point in the path
                if previous[current] != -1:
                    try:
                        # Calculate angle between previous->current and current->neighbor
                        angle = calculate_angle(
                            (df.iloc[previous[current]]['x'], df.iloc[previous[current]]['y']),
                            (df.iloc[current]['x'], df.iloc[current]['y']),
                            (df.iloc[neighbor]['x'], df.iloc[neighbor]['y'])
                        )
                        # Skip if the absolute angle is >= 90 degrees
                        if abs(angle) >= 90:
                            continue
                    except Exception as e:
                        print(f"Warning: Error calculating angle: {str(e)}")
                        continue

                # Step 8: Check for path crossings
                has_intersection = False
                # Traverse the current path from current point back to start
                path_point = previous[current]
                while path_point != -1 and previous[path_point] != -1:
                    # Check if new segment intersects with any existing path segment
                    if segments_intersect(
                            (df.iloc[path_point]['x'], df.iloc[path_point]['y']),
                            (df.iloc[previous[path_point]]['x'], df.iloc[previous[path_point]]['y']),
                            (df.iloc[current]['x'], df.iloc[current]['y']),
                            (df.iloc[neighbor]['x'], df.iloc[neighbor]['y'])
                    ):
                        has_intersection = True
                        break
                    path_point = previous[path_point]

                # Skip if there's an intersection
                if has_intersection:
                    continue

                # Step 9: Update distance if new path is better
                # For maximum value path, we add the new weight to current distance
                new_dist = distances[current] + weight
                if new_dist > distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current

        # Step 10: Find the end point with maximum value for this starting point
        end_idx = distances.index(max(distances))
        current_value = distances[end_idx]

        # Step 11: Update best solution if this path is better
        if current_value > best_value:
            best_value = current_value
            # Reconstruct the path by following previous pointers
            path = []
            current = end_idx
            while current != -1:
                path.append(current)
                current = previous[current]
            path.reverse()
            best_path = path
            print(f"Found better path with value: {best_value:.3f}")

    # Step 12: Return the best path and its value
    return best_path, best_value


def plot_path(df: pd.DataFrame, path_indices: List[int], title: str = "Optimal Path"):
    """
    Plot the points and the optimal path using matplotlib.

    Args:
        df: DataFrame with point data
        path_indices: Indices of points in the optimal path
        title: Plot title
    """
    plt.figure(figsize=(12, 8))  # Increased figure width to accommodate legend

    # Plot all points with color based on their value
    plt.scatter(df['x'], df['y'], c=df['value'], cmap='viridis', s=30)  # Reduced point size

    # Plot the optimal path if one exists
    if path_indices:
        # Extract points in the path from DataFrame
        path_df = df.iloc[path_indices]
        # Plot path as red line with reduced thickness
        plt.plot(path_df['x'], path_df['y'], 'r-', linewidth=0.75,
                 label='Optimal Path')  # Further reduced line thickness

        # Add arrows to show path direction with reduced size
        for i in range(len(path_indices) - 1):
            plt.arrow(path_df.iloc[i]['x'], path_df.iloc[i]['y'],
                      path_df.iloc[i + 1]['x'] - path_df.iloc[i]['x'],
                      path_df.iloc[i + 1]['y'] - path_df.iloc[i]['y'],
                      head_width=0.03, head_length=0.05,  # Reduced arrow size
                      fc='red', ec='red', alpha=0.5)

        # Label start and end points
        start_point = path_df.iloc[0]
        end_point = path_df.iloc[-1]

        # Plot start point in green and end point in red with reduced size
        plt.scatter(start_point['x'], start_point['y'], c='green', s=60, marker='*', label='Start')
        plt.scatter(end_point['x'], end_point['y'], c='red', s=60, marker='*', label='End')

        # Add text labels
        plt.annotate('Start', (start_point['x'], start_point['y']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold')
        plt.annotate('End', (end_point['x'], end_point['y']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold')

    # Add plot elements
    plt.colorbar(label='Point Value')  # Show color scale for point values
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Move legend to top left corner of the plot
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', borderaxespad=0.)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the pathfinding algorithm.
    Reads data, finds optimal path, and visualizes results.
    """
    try:
        print("Starting pathfinding algorithm...")

        # Read points from CSV file
        print("Reading data from CSV file...")
        df = read_points_from_csv('H:\\JOB Applications\\NL\\2025\\Interviews\\WhiteSpace\\'
                                  'Bahman_Askari_whitespace_codingchallenges_20200306\\3_random_routes.csv')

        # Find optimal path
        print("\nFinding optimal path...")
        optimal_path_indices, max_value = find_max_value_path(df)

        # Print results
        print("\nResults:")
        print(f"Maximum path value: {max_value:.3f}")
        print(f"Number of points in path: {len(optimal_path_indices)}")
        print("\nPoints in optimal path:")
        for i, idx in enumerate(optimal_path_indices):
            point = df.iloc[idx]  # Get point data from DataFrame
            print(f"Point {i + 1}: ({point['x']:.3f}, {point['y']:.3f}) - Value: {point['value']:.3f}")

        # Plot results
        print("\nGenerating plot...")
        plot_path(df, optimal_path_indices, f"Optimal Path (Value: {max_value:.3f})")
        print("Done!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()