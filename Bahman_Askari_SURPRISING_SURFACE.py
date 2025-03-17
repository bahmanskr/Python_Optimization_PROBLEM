"""
Surprising Surface - 3D Surface Area Optimization
-----------------------------------------------
This program analyzes a set of 3D points to find the surface area of the smallest
enclosing volume that efficiently captures the shape.

Input: JSON file containing 3D point coordinates (x, y, z)
Output: Surface area and visualization of the point set
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import pandas as pd
from typing import List, Tuple, Dict
import os
from scipy.spatial import ConvexHull, Delaunay


def read_points_from_json(filename: str) -> pd.DataFrame:
    """
    Read 3D points from JSON file.

    Args:
        filename: Path to JSON file containing point data

    Returns:
        DataFrame with columns x, y, z
    """
    try:
        # Read JSON file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Get points from the first key's value
        points = list(data.values())[0]

        # Convert points to DataFrame
        df = pd.DataFrame(points, columns=['x', 'y', 'z'])

        print(f"Loaded {len(df)} 3D points from {filename}")
        return df

    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        raise


def calculate_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the area of a triangle given its three vertices.

    Args:
        p1, p2, p3: 3D coordinates of triangle vertices

    Returns:
        float: Area of the triangle
    """
    # Calculate two edges of the triangle
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate area using cross product
    cross_product = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross_product)

    return area


def find_enclosing_surface(points: np.ndarray, alpha: float = None) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """
    Find the smallest enclosing surface using alpha shapes algorithm.

    Args:
        points: Nx3 array of 3D points
        alpha: Parameter controlling the level of detail (default=None for auto-calculation)

    Returns:
        Tuple containing:
        - Array of triangles forming the surface
        - Total surface area
        - List of triangle vertices
    """
    print("\nFinding enclosing surface...")

    # Create Delaunay triangulation
    tri = Delaunay(points)

    # Calculate a reasonable alpha value if not provided
    if alpha is None:
        # Calculate the average distance between points
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distances.append(np.linalg.norm(points[i] - points[j]))
        alpha = np.mean(distances) * 0.5  # Use half the average distance
        print(f"Using auto-calculated alpha value: {alpha:.2f}")

    # Initialize lists for valid triangles and their areas
    valid_triangles = []
    triangle_vertices = []
    total_area = 0

    print("Analyzing triangles...")
    print(f"Total number of triangles: {len(tri.simplices)}")

    # Process each triangle in the triangulation
    rejected_count = 0
    for simplex in tri.simplices:
        # Get the three points forming this triangle
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]

        # Calculate triangle properties
        area = calculate_triangle_area(p1, p2, p3)

        # Calculate circumradius
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2  # semi-perimeter

        try:
            # Calculate circumradius using the formula R = abc/(4A)
            if area > 0:
                circumradius = (a * b * c) / (4 * area)

                # Keep triangle if its circumradius is less than alpha
                if circumradius < alpha:
                    valid_triangles.append(simplex)
                    triangle_vertices.append([p1, p2, p3])
                    total_area += area
                else:
                    rejected_count += 1
        except:
            rejected_count += 1
            continue

    print(f"Found {len(valid_triangles)} valid triangles")
    print(f"Rejected {rejected_count} triangles")
    print(f"Total surface area: {total_area:.2f} square units")

    # If no valid triangles found, try with a larger alpha
    if len(valid_triangles) == 0:
        print("\nNo valid triangles found. Trying with larger alpha value...")
        return find_enclosing_surface(points, alpha * 2)

    return np.array(valid_triangles), total_area, triangle_vertices


def plot_3d_points(df: pd.DataFrame, triangles: List[np.ndarray], title: str = "3D Point Set"):
    """
    Create an interactive 3D visualization of the point set and surface.

    Args:
        df: DataFrame with x, y, z coordinates
        triangles: List of triangle vertices
        title: Plot title
    """
    print("\nGenerating 3D visualization...")

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    scatter = ax.scatter(df['x'], df['y'], df['z'],
                         c=df['z'],  # Color points by z-coordinate
                         cmap='viridis',
                         s=20,  # Point size
                         alpha=0.6,
                         label='Points')

    # Plot surface triangles
    surface = Poly3DCollection(triangles, alpha=0.3)
    surface.set_facecolor('red')
    ax.add_collection3d(surface)

    # Add colorbar
    plt.colorbar(scatter, label='Z Coordinate')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    # Add grid
    ax.grid(True)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_raw_points(df: pd.DataFrame, title: str = "Raw 3D Point Cloud"):
    """
    Create an interactive 3D visualization of just the point cloud.

    Args:
        df: DataFrame with x, y, z coordinates
        title: Plot title
    """
    print("\nGenerating raw point cloud visualization...")

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with different colors and sizes for better visibility
    scatter = ax.scatter(df['x'], df['y'], df['z'],
                         c=df['z'],  # Color points by z-coordinate
                         cmap='viridis',
                         s=50,  # Larger point size for better visibility
                         alpha=0.8,
                         label='Points')

    # Add colorbar
    plt.colorbar(scatter, label='Z Coordinate')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    # Add grid
    ax.grid(True)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to analyze the 3D point set and find enclosing surface.
    """
    try:
        print("Starting 3D point set analysis...")

        # Read points from JSON file
        print("\nReading data from JSON file...")
        df = read_points_from_json('H:\\JOB Applications\\NL\\2025\\Interviews\\WhiteSpace\\'
                                   'Bahman_Askari_whitespace_codingchallenges_20200306\\4_surprising_surface.json')

        # First, visualize the raw point cloud
        print("\nVisualizing raw point cloud...")
        plot_raw_points(df, "Raw 3D Point Cloud")

        # Convert DataFrame to numpy array
        points = df[['x', 'y', 'z']].values

        # Find enclosing surface
        triangles, surface_area, triangle_vertices = find_enclosing_surface(points)

        # Print detailed results
        print("\nResults:")
        print(f"Number of input points: {len(points)}")
        print(f"Number of surface triangles: {len(triangles)}")
        print(f"Total surface area: {surface_area:.2f} square units")

        # Create 3D visualization with surface
        print("\nVisualizing point cloud with enclosing surface...")
        plot_3d_points(df, triangle_vertices,
                       f"3D Point Set with Enclosing Surface\nSurface Area: {surface_area:.2f} square units")

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()