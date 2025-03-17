# Python_Optimization_PROBLEMS

## Nuanced Navigation

The problem requires designing an optimal cyclic route for a sailing vessel to visit 12 sites while considering navigation and speed constraints. The vessel cannot sail directly against the northward wind, restricting south-to-north movement. The speed decreases based on the number of passengers, starting with 10 passengers, an additional 10 board at each site, with no disembarkation. The objective is to determine the fastest possible route that visits all sites once while adhering to these constraints.

## TRICKY TRACKS
This challenge aims to solve a railway and subway track optimization problem where:
- We have multiple railway and subway tracks
- Tracks are defined by start and end coordinates
- Railway tracks cannot intersect with other railway tracks
- Subway tracks cannot intersect with other subway tracks
- Railway-subway intersections are allowed but require stations
- Goal: Maximize the number of usable tracks given a limited number of stations

## RANDOM ROUTES
The Random Routes problem involves finding a maximum-value path in a 2D point set with specific geometric constraints. Each point in the set has a value between 0 and 1, and the path must satisfy the following requirements:
- Must be continuous and non-crossing
- Each point can be visited at most once
- Angles between consecutive segments must be between -90° and +90° (abs(Angle) <=90 ).
- Path can start and end at any different points
- The goal is to maximize the cumulative value of points in the path


## SURPRISING SURFACE
The goal of the problem is to analyze a set of 3D points to find the surface area of the smallest enclosing volume that efficiently captures the shape of the point cloud. The proposed algorithm should create a surface that balances between overfitting and underfitting.
