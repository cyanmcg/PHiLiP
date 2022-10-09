#include "eikonal_cube.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_bounded_cube.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// DISTANCE EVALUATION IN BOUNDED CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
EikonalCube<dim, nstate>::EikonalCube(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> EikonalCube<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
    );
    Grids::straight_bounded_cube<dim,Triangulation>(grid, domain_left, domain_right, number_of_cells_per_direction);

    return grid;
}

template <int dim, int nstate>
void EikonalCube<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    const std::string grid_type_string = "straight_bounded_cube";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << this->domain_left << std::endl;
    this->pcout << "- - Domain right: " << this->domain_right << std::endl;
    this->pcout << "- - Number of cells in each direction: " << this->number_of_cells_per_direction << std::endl;
    if constexpr(dim==2) this->pcout << "- - Domain area: " << this->domain_size << std::endl;
    //if constexpr(dim==3) this->pcout << "- - Domain volume: " << this->domain_size << std::endl;
}

template class EikonalCube <PHILIP_DIM,1>;

} // FlowSolver namespace
} // PHiLiP namespace