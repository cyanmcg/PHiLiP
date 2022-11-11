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

    //std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
    //dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
    //dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
    //grid->add_periodicity(matched_pairs);

    grid->refine_global(number_of_refinements);

    //for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
    //    for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
    //        // specify BC to different faces
    //        if (cell->face(face)->at_boundary()) {
    //            unsigned int current_id = cell->face(face)->boundary_id();
    //            if (current_id == 2) cell->face(face)->set_boundary_id (1001); // wall
    //            if (current_id == 3) cell->face(face)->set_boundary_id (1005); // farfield
    //        }
    //    }
    //}

    for (auto cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                // all faces are wall BC
                if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id (1001);

                // specify BC to different faces
                //if (cell->face(face)->at_boundary()) {
                //    unsigned int current_id = cell->face(face)->boundary_id();
                //    if (current_id == 0 || current_id == 1 || current_id == 2) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                //    if (current_id == 3) cell->face(face)->set_boundary_id (1005); // farfield
                //}
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