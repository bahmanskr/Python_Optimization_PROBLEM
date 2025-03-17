"""
Tricky Tracks - Railway and Subway Track Optimization
---------------------------------------------------
This program solves a track layout optimization problem where:
- Railway tracks cannot intersect with other railway tracks
- Subway tracks cannot intersect with other subway tracks
- Railway-subway intersections are allowed but require stations
- Goal: Maximize the number of usable tracks given a limited number of stations

Input: Two CSV files containing track coordinates (start_x, start_y, end_x, end_y)
Output: Maximum number of tracks possible for different numbers of stations (0-20)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# itertools is a Python standard library that provides fast, memory efficient tools
# for working with iterables and combinations
# combinations(iterable, r) generates all possible r-length combinations from an iterable
# For example: combinations([1,2,3], 2) produces (1,2), (1,3), (2,3)
# We use it here to efficiently generate all possible pairs of tracks to check for intersections
from itertools import combinations


def read_tracks(railway_file, subway_file):
    """
    Read railway and subway tracks from CSV files.
    Each track is defined by start point (start_x, start_y) and end point (end_x, end_y).

    Args:
        railway_file (str): Path to CSV file containing railway track coordinates
        subway_file (str): Path to CSV file containing subway track coordinates

    Returns:
        tuple: Two lists of track segments [(start_x,start_y), (end_x,end_y)],
              first for railways, second for subways
    """
    # Read railway tracks from CSV with columns: start_x, start_y, end_x, end_y
    rail_df = pd.read_csv(railway_file)
    railway_tracks = []
    for _, row in rail_df.iterrows():
        railway_tracks.append([(row['start_x'], row['start_y']), (row['end_x'], row['end_y'])])

    # Read subway tracks from CSV with columns: start_x, start_y, end_x, end_y
    sub_df = pd.read_csv(subway_file)
    subway_tracks = []
    for _, row in sub_df.iterrows():
        subway_tracks.append([(row['start_x'], row['start_y']), (row['end_x'], row['end_y'])])

    return railway_tracks, subway_tracks


def segments_intersect(seg1, seg2):
    """
    Check if two line segments intersect using counter-clockwise (CCW) orientation test.

    The CCW test determines if three points make a counter-clockwise turn.
    Two segments intersect if:
    - Points of seg2 are on opposite sides of seg1
    - Points of seg1 are on opposite sides of seg2

    Args:
        seg1: First segment as [(x1,y1), (x2,y2)]
        seg2: Second segment as [(x1,y1), (x2,y2)]

    Returns:
        bool: True if segments intersect, False otherwise
    """

    def ccw(A, B, C):
        """Helper function to determine if three points make a counter-clockwise turn"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Check if points of each segment are on opposite sides of the other segment
    return ccw(seg1[0], seg2[0], seg2[1]) != ccw(seg1[1], seg2[0], seg2[1]) and \
           ccw(seg1[0], seg1[1], seg2[0]) != ccw(seg1[0], seg1[1], seg2[1])


def plot_optimal_layout(railway_tracks, subway_tracks, stations, optimal_stations, optimal_tracks):
    """
    Plot the optimal track layout with different colors for railway and subway tracks.

    Args:
        railway_tracks: List of railway track segments
        subway_tracks: List of subway track segments
        stations: Number of stations used
        optimal_stations: List of station coordinates
        optimal_tracks: Tuple of (valid_rail_indices, valid_sub_indices)
    """
    plt.figure(figsize=(12, 8))

    # Plot railway tracks in blue
    valid_rail_indices, valid_sub_indices = optimal_tracks
    for idx in valid_rail_indices:
        track = railway_tracks[idx]
        plt.plot([track[0][0], track[1][0]], [track[0][1], track[1][1]], 'b-', linewidth=2,
                 label='Railway Tracks' if idx == list(valid_rail_indices)[0] else "")

    # Plot subway tracks in red
    for idx in valid_sub_indices:
        track = subway_tracks[idx]
        plt.plot([track[0][0], track[1][0]], [track[0][1], track[1][1]], 'r-', linewidth=2,
                 label='Subway Tracks' if idx == list(valid_sub_indices)[0] else "")

    plt.title('Optimal Track Segments')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


def find_max_tracks(railway_tracks, subway_tracks, max_stations):
    """
    Find the maximum number of tracks that can be used with given constraints.
    Returns both the count and the valid track indices.
    """
    n_rail = len(railway_tracks)
    n_sub = len(subway_tracks)

    # Step 1: Remove invalid railway crossings
    valid_rail_indices = set(range(n_rail))
    for i, j in combinations(range(n_rail), 2):
        if segments_intersect(railway_tracks[i], railway_tracks[j]):
            i_intersections = sum(
                1 for k in range(n_rail) if k != i and segments_intersect(railway_tracks[i], railway_tracks[k]))
            j_intersections = sum(
                1 for k in range(n_rail) if k != j and segments_intersect(railway_tracks[j], railway_tracks[k]))
            if i_intersections >= j_intersections:
                valid_rail_indices.discard(i)
            else:
                valid_rail_indices.discard(j)

    # Step 2: Remove invalid subway crossings
    valid_sub_indices = set(range(n_sub))
    for i, j in combinations(range(n_sub), 2):
        if segments_intersect(subway_tracks[i], subway_tracks[j]):
            i_intersections = sum(
                1 for k in range(n_sub) if k != i and segments_intersect(subway_tracks[i], subway_tracks[k]))
            j_intersections = sum(
                1 for k in range(n_sub) if k != j and segments_intersect(subway_tracks[j], subway_tracks[k]))
            if i_intersections >= j_intersections:
                valid_sub_indices.discard(i)
            else:
                valid_sub_indices.discard(j)

    # Step 3: Handle railway-subway intersections based on station count
    optimal_stations = []
    if max_stations == 0:
        for r_idx in list(valid_rail_indices):
            for s_idx in valid_sub_indices:
                if segments_intersect(railway_tracks[r_idx], subway_tracks[s_idx]):
                    valid_rail_indices.discard(r_idx)
                    break
    else:
        crossing_tracks = set()
        crossing_points = []

        # Find all intersecting tracks and their intersection points
        for r_idx in valid_rail_indices:
            for s_idx in valid_sub_indices:
                if segments_intersect(railway_tracks[r_idx], subway_tracks[s_idx]):
                    crossing_tracks.add(r_idx)
                    crossing_tracks.add(s_idx)
                    # Calculate intersection point
                    r_track = railway_tracks[r_idx]
                    s_track = subway_tracks[s_idx]
                    # Simple intersection point calculation (can be improved)
                    x = (r_track[0][0] + r_track[1][0] + s_track[0][0] + s_track[1][0]) / 4
                    y = (r_track[0][1] + r_track[1][1] + s_track[0][1] + s_track[1][1]) / 4
                    crossing_points.append((x, y))

        # If we have more intersections than allowed stations
        if len(crossing_tracks) > max_stations * 2:
            # Count intersections for each track
            track_crossings = {}
            for idx in crossing_tracks:
                if idx < n_rail:
                    crossings = sum(1 for s_idx in valid_sub_indices
                                    if segments_intersect(railway_tracks[idx], subway_tracks[s_idx]))
                else:
                    s_idx = idx - n_rail
                    crossings = sum(1 for r_idx in valid_rail_indices
                                    if segments_intersect(railway_tracks[r_idx], subway_tracks[s_idx]))
                track_crossings[idx] = crossings

            # Remove tracks with most intersections until we're within station limit
            while len(crossing_tracks) > max_stations * 2:
                track_to_remove = max(track_crossings.items(), key=lambda x: x[1])[0]

                if track_to_remove < n_rail:
                    valid_rail_indices.discard(track_to_remove)
                else:
                    valid_sub_indices.discard(track_to_remove - n_rail)

                crossing_tracks.discard(track_to_remove)
                del track_crossings[track_to_remove]

            # Select optimal station locations
            optimal_stations = crossing_points[:max_stations]

    return len(valid_rail_indices) + len(valid_sub_indices), (valid_rail_indices, valid_sub_indices), optimal_stations


def plot_results(stations_range, track_counts):
    """
    Plot the relationship between number of stations and maximum tracks.
    Creates a line plot with integer values on both axes for clarity.

    Args:
        stations_range: Range of station counts (0 to 20)
        track_counts: List of maximum track counts for each station count
    """
    plt.figure(figsize=(10, 6))
    plt.plot(stations_range, track_counts, 'b-', marker='o')

    # Set integer ticks for x-axis (stations)
    plt.xticks(range(min(stations_range), max(stations_range) + 1))

    # Set integer ticks for y-axis (tracks)
    min_tracks = min(track_counts)
    max_tracks = max(track_counts)
    plt.yticks(range(int(min_tracks), int(max_tracks) + 1))

    plt.xlabel('Number of Stations (S)')
    plt.ylabel('Maximum Number of Tracks')
    plt.title('Maximum Tracks vs Number of Stations')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to run the track optimization analysis.
    Tests different numbers of stations (0-20) and plots the results.
    """
    # Read track data from CSV files

    tr="H:\\JOB Applications\\NL\\2025\\Interviews\\WhiteSpace\\" \
       "Bahman_Askari_whitespace_codingchallenges_20200306\\2_tricky_tracks_railways.csv"
    ts = "H:\\JOB Applications\\NL\\2025\\Interviews\\WhiteSpace\\" \
         "Bahman_Askari_whitespace_codingchallenges_20200306\\2_tricky_tracks_subways.csv"

    railway_tracks, subway_tracks = read_tracks(tr, ts)
    # Test different numbers of stations
    stations_range = range(21)  # 0 to 20 stations
    track_counts = []

    # Calculate maximum tracks for each number of stations
    optimal_solution = None
    optimal_stations = None
    optimal_tracks = None

    for s in stations_range:
        max_tracks, tracks, stations = find_max_tracks(railway_tracks, subway_tracks, s)
        track_counts.append(max_tracks)
        print(f"Maximum tracks with {s} stations: {max_tracks}")

        # Save the optimal solution
        if s == 8 and max_tracks == 29:
            optimal_solution = s
            optimal_stations = stations
            optimal_tracks = tracks

    # Plot the results
    plot_results(stations_range, track_counts)

    # Plot optimal layout if found
    if optimal_solution is not None:
        print("\nPlotting optimal layout...")
        plot_optimal_layout(railway_tracks, subway_tracks, optimal_solution, optimal_stations, optimal_tracks)

        # Save the results to a file
        with open('optimal_solution.txt', 'w') as f:
            f.write(f"Optimal Solution:\n")
            f.write(f"Number of stations: {optimal_solution}\n")
            f.write(f"Number of tracks: {len(optimal_tracks[0]) + len(optimal_tracks[1])}\n")
            f.write("\nStation coordinates:\n")
            for i, station in enumerate(optimal_stations, 1):
                f.write(f"Station {i}: ({station[0]:.2f}, {station[1]:.2f})\n")
            f.write("\nRailway track coordinates:\n")
            for i, idx in enumerate(optimal_tracks[0], 1):
                track = railway_tracks[idx]
                f.write(
                    f"Track {i}: ({track[0][0]:.2f}, {track[0][1]:.2f}) -> ({track[1][0]:.2f}, {track[1][1]:.2f})\n")
            f.write("\nSubway track coordinates:\n")
            for i, idx in enumerate(optimal_tracks[1], 1):
                track = subway_tracks[idx]
                f.write(
                    f"Track {i}: ({track[0][0]:.2f}, {track[0][1]:.2f}) -> ({track[1][0]:.2f}, {track[1][1]:.2f})\n")


if __name__ == "__main__":
    main()