#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <stdlib.h>
#include <iostream>

#include "straight_bounded_cube.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void straight_bounded_cube(std::shared_ptr<TriangulationType> &grid,
                            const double domain_left,
                            const double domain_right,
                            const int number_of_cells_per_direction)
{
    // Get equivalent number of refinements
    const int number_of_refinements = log(number_of_cells_per_direction)/log(2);

    // Definition for each type of grid
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1001);
            }
        }
}

#if PHILIP_DIM==1
    template void straight_bounded_cube<PHILIP_DIM, dealii::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif
#if PHILIP_DIM!=1
    template void straight_bounded_cube<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif

} // namespace Grids
} // namespace PHiLiP