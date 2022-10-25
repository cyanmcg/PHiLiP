#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "physics.h"
#include "eikonal.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Eikonal Class
//================================================================
template <int dim, int nstate, typename real>
Eikonal<dim,nstate,real>::Eikonal ( 
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(has_nonzero_physical_source,manufactured_solution_function)
{
    static_assert(nstate==1, "Physics::Eikonal() should be created with nstate==dim");
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Eikonal<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &conservative_soln) const
{
    return convective_flux_templated<real>(conservative_soln);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> Eikonal<dim,nstate,real>
::convective_flux_templated (
    const std::array<real2,nstate> &conservative_soln) const
{
    (void) conservative_soln;
    std::array<dealii::Tensor<1,dim,real2>,nstate> conv_flux;
    // No convective flux for Eikonal
    for (int flux_dim=0; flux_dim<nstate; ++flux_dim) {
        conv_flux[flux_dim] = 0.0;
    }
    return conv_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Eikonal<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    return dissipative_flux_templated<real>(conservative_soln,solution_gradient,cell_index);
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
template <typename real2>
std::array<dealii::Tensor<1,dim,real2>,nstate> Eikonal<dim,nstate,real>
::dissipative_flux_templated (
    const std::array<real2,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    (void) cell_index;
    std::array<dealii::Tensor<1,dim,real2>,nstate> diss_flux;

    for (int flux_dim=0; flux_dim<nstate; ++flux_dim) {
        for (int d=0; d<dim; ++d){
            diss_flux[flux_dim][d] = conservative_soln[flux_dim]*solution_gradient[flux_dim][d];
        }
    }

    return diss_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::physical_source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const std::array<dealii::Tensor<2,dim,real>,nstate> &solution_hessian,
    const dealii::types::global_dof_index cell_index) const
{
    (void) pos;
    (void) cell_index;
    (void) solution_gradient;
    std::array<real,nstate> physical_source;
    for (int i=0; i<nstate; i++) {
        physical_source[i] = 0.0;
        for (int j=0;j<dim;j++){
            physical_source[i] += solution_hessian[i][j][j];
        }
        physical_source[i] = 1.0+conservative_soln[i]*physical_source[i];
    }
    return physical_source;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::types::global_dof_index cell_index) const
{
    std::array<real,nstate> conv_source_term = convective_source_term(pos);
    std::array<real,nstate> diss_source_term = dissipative_source_term(pos,cell_index);
    std::array<real,nstate> phys_source_source_term = physical_source_source_term(pos,cell_index);
    std::array<real,nstate> source_term;
    for (int s=0; s<nstate; s++)
    {
        source_term[s] = conv_source_term[s] + diss_source_term[s] - phys_source_source_term[s];
    }
    return source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::get_manufactured_solution_value (
    const dealii::Point<dim,real> &pos) const
{
    std::array<real,nstate> manufactured_solution;
    for (int s=0; s<nstate; s++) {
        manufactured_solution[s] = this->manufactured_solution_function->value (pos, s);
        if (s==0) {
            assert(manufactured_solution[s] > 0);
        }
    }
    return manufactured_solution;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Eikonal<dim,nstate,real>
::get_manufactured_solution_gradient (
    const dealii::Point<dim,real> &pos) const
{
    std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
    this->manufactured_solution_function->vector_gradient(pos,manufactured_solution_gradient_dealii);
    std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient;
    for (int d=0;d<dim;d++) {
        for (int s=0; s<nstate; s++) {
            manufactured_solution_gradient[s][d] = manufactured_solution_gradient_dealii[s][d];
        }
    }
    return manufactured_solution_gradient;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<2,dim,real>,nstate> Eikonal<dim,nstate,real>
::get_manufactured_solution_hessian (
    const dealii::Point<dim,real> &pos) const
{
    std::vector<dealii::Tensor<2,dim,real>> manufactured_solution_hessian_dealii(nstate);
    this->manufactured_solution_function->matrix_hessian(pos,manufactured_solution_hessian_dealii);
    std::array<dealii::Tensor<2,dim,real>,nstate> manufactured_solution_hessian;
    for (int s=0; s<nstate; s++) {
        for (int d1=0;d1<dim;d1++) {
            for (int d2=0;d2<dim;d2++) {
                manufactured_solution_hessian[s][d1][d2] = manufactured_solution_hessian_dealii[s][d1][d2];
            }
        }
    }
    return manufactured_solution_hessian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::convective_source_term (
    const dealii::Point<dim,real> &pos) const
{
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos);

    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos);

    dealii::Tensor<1,nstate,real> convective_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = convective_flux_directional_jacobian(manufactured_solution, normal);

        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d];
            }
            convective_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> convective_source_term;
    for (int s=0; s<nstate; s++) {
        convective_source_term[s] = convective_flux_divergence[s];
    }

    return convective_source_term;
}

// Returns the value from a CoDiPack or Sacado variable.
template<typename real>
double getValue(const real &x) {
    if constexpr(std::is_same<real,double>::value) {
        return x;
    }
    else if constexpr(std::is_same<real,FadType>::value) {
        return x.val(); // sacado
    } 
    else if constexpr(std::is_same<real,FadFadType>::value) {
        return x.val().val(); // sacado
    }
    else if constexpr(std::is_same<real,RadType>::value) {
      return x.value(); // CoDiPack
    } 
    else if(std::is_same<real,RadFadType>::value) {
        return x.value().value(); // CoDiPack
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> Eikonal<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
    }

    // Compute AD convective flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_convective_flux = convective_flux_templated<adtype>(AD_conservative_soln);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_convective_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::dissipative_source_term (
    const dealii::Point<dim,real> &pos,
    const dealii::types::global_dof_index cell_index) const
{    
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos); // from Euler
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos); // from Euler
    
    // Get Manufactured Solution hessian
    std::array<dealii::SymmetricTensor<2,dim,real>,nstate> manufactured_solution_hessian;
    for (int s=0; s<nstate; s++) {
        dealii::SymmetricTensor<2,dim,real> hessian = this->manufactured_solution_function->hessian(pos,s);
        for (int dr=0;dr<dim;dr++) {
            for (int dc=0;dc<dim;dc++) {
                manufactured_solution_hessian[s][dr][dc] = hessian[dr][dc];
            }
        }
    }

    // First term -- wrt to the conservative variables
    // This is similar, should simply provide this function a flux_directional_jacobian() -- could restructure later
    dealii::Tensor<1,nstate,real> dissipative_flux_divergence;
    for (int d=0;d<dim;d++) {
        dealii::Tensor<1,dim,real> normal;
        normal[d] = 1.0;
        const dealii::Tensor<2,nstate,real> jacobian = dissipative_flux_directional_jacobian(manufactured_solution, manufactured_solution_gradient, normal, cell_index);
        
        // get the directional jacobian wrt gradient
        std::array<dealii::Tensor<2,nstate,real>,dim> jacobian_wrt_gradient;
        for (int d_gradient=0;d_gradient<dim;d_gradient++) {
            
            // get the directional jacobian wrt gradient component (x,y,z)
            const dealii::Tensor<2,nstate,real> jacobian_wrt_gradient_component = dissipative_flux_directional_jacobian_wrt_gradient_component(manufactured_solution, manufactured_solution_gradient, normal, d_gradient, cell_index);
            
            // store each component in jacobian_wrt_gradient -- could do this in the function used above
            for (int sr = 0; sr < nstate; ++sr) {
                for (int sc = 0; sc < nstate; ++sc) {
                    jacobian_wrt_gradient[d_gradient][sr][sc] = jacobian_wrt_gradient_component[sr][sc];
                }
            }
        }

        //dissipative_flux_divergence += jacobian*manufactured_solution_gradient[d]; <-- needs second term! (jac wrt gradient)
        for (int sr = 0; sr < nstate; ++sr) {
            real jac_grad_row = 0.0;
            for (int sc = 0; sc < nstate; ++sc) {
                jac_grad_row += jacobian[sr][sc]*manufactured_solution_gradient[sc][d]; // Euler is the same as this
                // Second term -- wrt to the gradient of conservative variables
                // -- add the contribution of each gradient component (e.g. x,y,z for dim==3)
                for (int d_gradient=0;d_gradient<dim;d_gradient++) {
                    jac_grad_row += jacobian_wrt_gradient[d_gradient][sr][sc]*manufactured_solution_hessian[sc][d_gradient][d]; // symmetric so d indexing works both ways
                }
            }
            dissipative_flux_divergence[sr] += jac_grad_row;
        }
    }
    std::array<real,nstate> dissipative_source_term;
    for (int s=0; s<nstate; s++) {
        dissipative_source_term[s] = dissipative_flux_divergence[s];
    }

    return dissipative_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> Eikonal<dim,nstate,real>
::dissipative_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal,
    const dealii::types::global_dof_index cell_index) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; s++) {
        adtype ADvar(nstate, s, getValue<real>(conservative_soln[s])); // create AD variable
        AD_conservative_soln[s] = ADvar;
        for (int d=0;d<dim;d++) {
            AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient, cell_index);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::Tensor<2,nstate,real> Eikonal<dim,nstate,real>
::dissipative_flux_directional_jacobian_wrt_gradient_component (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::Tensor<1,dim,real> &normal,
    const int d_gradient,
    const dealii::types::global_dof_index cell_index) const
{
    using adtype = FadType;

    // Initialize AD objects
    std::array<adtype,nstate> AD_conservative_soln;
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_solution_gradient;
    for (int s=0; s<nstate; s++) {
        AD_conservative_soln[s] = getValue<real>(conservative_soln[s]);
        for (int d=0;d<dim;d++) {
            if(d == d_gradient){
                adtype ADvar(nstate, s, getValue<real>(solution_gradient[s][d])); // create AD variable
                AD_solution_gradient[s][d] = ADvar;
            }
            else {
                AD_solution_gradient[s][d] = getValue<real>(solution_gradient[s][d]);
            }
        }
    }

    // Compute AD dissipative flux
    std::array<dealii::Tensor<1,dim,adtype>,nstate> AD_dissipative_flux = dissipative_flux_templated<adtype>(AD_conservative_soln, AD_solution_gradient, cell_index);

    // Assemble the directional Jacobian
    dealii::Tensor<2,nstate,real> jacobian;
    for (int sp=0; sp<nstate; sp++) {
        // for each perturbed state (sp) variable
        for (int s=0; s<nstate; s++) {
            jacobian[s][sp] = 0.0;
            for (int d=0;d<dim;d++) {
                // Compute directional jacobian
                jacobian[s][sp] += AD_dissipative_flux[s][d].dx(sp)*normal[d];
            }
        }
    }
    return jacobian;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::physical_source_source_term (
    const dealii::Point<dim,real> &pos,
    const dealii::types::global_dof_index cell_index) const
{    
    // Get Manufactured Solution values
    const std::array<real,nstate> manufactured_solution = get_manufactured_solution_value(pos); // from Euler
    
    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient = get_manufactured_solution_gradient(pos); // from Euler

    // Get Manufactured Solution gradient
    const std::array<dealii::Tensor<2,dim,real>,nstate> manufactured_solution_hessian = get_manufactured_solution_hessian(pos); // from Euler
    
    std::array<real,nstate> physical_source_source_term;
    for (int i=0;i<nstate;i++){
        physical_source_source_term = physical_source_term(pos, manufactured_solution, manufactured_solution_gradient, manufactured_solution_hessian, cell_index);
    }

    return physical_source_source_term;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Eikonal<dim, nstate, real>
::convective_numerical_split_flux(const std::array<real,nstate> &/*conservative_soln1*/,
                                  const std::array<real,nstate> &/*conservative_soln2*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    
    for (int i=0;i<nstate;i++){
        conv_num_split_flux[i] = 0.0;
    }

    return conv_num_split_flux;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> Eikonal<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*conservative_soln*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    std::array<real,nstate> eig;
    for (int i=0; i<nstate; i++) {
        eig[i] = 0.0;
    }
    return eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
real Eikonal<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    const real max_eig = 0.0;

    return max_eig;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void Eikonal<dim,nstate,real>
::boundary_wall (
   const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    for (int istate=0; istate<nstate; ++istate) {
        soln_bc[istate] = 0.0;
        soln_grad_bc[istate] = 0.0;
        //soln_grad_bc[istate] = soln_grad_int[istate];
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void Eikonal<dim,nstate,real>
::boundary_farfield (
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &/*soln_bc*/,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
   for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
void Eikonal<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &/*pos*/,
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &/*soln_int*/,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1000) {
        // Manufactured solution boundary condition
        std::cout << "Manufactured solution boundary condition is not implemented!" << std::endl;
        std::abort();
    } 
    else if (boundary_type == 1001) {
        // Wall boundary condition 
        boundary_wall (soln_grad_int,soln_bc,soln_grad_bc);
    } 
    else if (boundary_type == 1002) {
        // Pressure outflow boundary condition 
        std::cout << "Pressure outflow boundary condition is not implemented!" << std::endl;
        std::abort();
    } 
    else if (boundary_type == 1003) {
        // Inflow boundary condition
        std::cout << "Inflow boundary condition is not implemented!" << std::endl;
        std::abort();
    } 
    else if (boundary_type == 1004) {
        // Riemann-based farfield boundary condition
        std::cout << "Riemann boundary condition is not implemented!" << std::endl;
        std::abort();
    } 
    else if (boundary_type == 1005) {
        // Simple farfield boundary condition
        boundary_farfield(soln_grad_int,soln_bc,soln_grad_bc);
    } 
    else if (boundary_type == 1006) {
        // Slip wall boundary condition
        std::cout << "Slip wall boundary condition is not implemented!" << std::endl;
        std::abort();
    } 
    else {
        std::cout << "Invalid boundary_type: " << boundary_type << std::endl;
        std::abort();
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::vector<std::string> Eikonal<dim,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names;
    for (unsigned int s=0; s<nstate; ++s) {
        names.push_back("minimum_wall_distance");
    }
    return names;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
dealii::UpdateFlags Eikonal<dim,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values
           | dealii::update_quadrature_points
           ;
}

// Instantiate explicitly
template class Eikonal < PHILIP_DIM, 1, double     >;
template class Eikonal < PHILIP_DIM, 1, FadType    >;
template class Eikonal < PHILIP_DIM, 1, RadType    >;
template class Eikonal < PHILIP_DIM, 1, FadFadType >;
template class Eikonal < PHILIP_DIM, 1, RadFadType >;
//==============================================================================

} // Physics namespace
} // PHiLiP namespace