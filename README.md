<p align="center" style="text-align:center">
<img src="https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/hubble_emitting.jpg" alt="drawing" width="300"/>
</p>

# Ananke
Generalized aerospace wrappers and optimization routines.

_Built with Python 3.6 in the Anaconda Python distribution._


## Modules List
| Module | Description |
| - | - |
| `ananke.frames` | Frame conversion and transformation tools. |
| `ananke.orbit` | Keplerian orbital calculation tools. |
| `ananke.opt` | Optimization toolkit. |
| `ananke.planets` | Planetary information. |
| `ananke.util` | Generic math capability and utilities. |
| `ananke.examples` | List of examples that use the above toolkits. |


## Current List of Examples
| Example | Description | 
| - | - |
| `ananke.examples.run_problem1()` | Trapezoidal collocation of a 1-D minimum control problem. |
| `ananke.examples.run_problem2()` | Trapezoidal collocation of a 2-D minimum control lunar lander trajectory. |
| `ananke.examples.run_problem3()` | Trapezoidal collocation of a 3-D spherical gravity rotating planet lander. |
| `ananke.examples.run_problem4()` | Trapezoidal collocation of a 2-D lander with minimum-fuel (NOT minimum-control). |
| `ananke.examples.run_problem5()` | Multi-phase 3-D lander trapezoidal collocation optimizer with multiple fitness functions. |
| `ananke.examples.run_problem6()` | Multi-phase 3-D lander Hermite-Simpson collocation optimizer with multiple fitness functions. |


## Interfaced Non-Default Toolkits
| Toolkit | Version | Install Command | Website |
| - | - | - | - |
| `numpy` | >= 1.17.4 | | |
| `scipy` | >= 1.3.1 | | |
| `pygmo` | >= 2.11.4 | `conda install -c conda-forge pygmo` | https://esa.github.io/pagmo2/ |
| `pykep` | >= 2.4.1 | `conda install -c conda-forge pykep` | https://esa.github.io/pykep/ |
| `matplotlib` | - | - | - |
| `numba` | - | - | - |





