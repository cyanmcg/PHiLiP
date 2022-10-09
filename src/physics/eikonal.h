#ifndef __EIKONAL__
#define __EIKONAL__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
class Eikonal : public PhysicsBase <dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
    using PhysicsBase<dim,nstate,real>::physical_source_term;
public:
    /// Constructor
    Eikonal (
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const bool                                                has_nonzero_physical_source = true);

    /// Destructor
    ~Eikonal() {};

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative flux: 0
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective flux Jacobian: \f$ \frac{\partial \mathbf{F}_{conv}}{\partial w} \cdot \mathbf{n} \f$
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Physical source term 
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective flux contribution to the source term
    std::array<real,nstate> convective_source_term (
        const dealii::Point<dim,real> &pos) const;

    /// Dissipative flux contribution to the source term
    std::array<real,nstate> dissipative_source_term (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const dealii::types::global_dof_index cell_index) const;

    /** Dissipative flux Jacobian wrt gradient component (repeated from NavierStokes)
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient,
        const dealii::types::global_dof_index cell_index) const;

    std::array<real,nstate> physical_source_source_term (
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Boundary condition handler
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
    
    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities
    std::vector<std::string> post_get_names () const override;
    
    /// For post processing purposes (update comment later)
    dealii::UpdateFlags post_get_needed_update_flags () const override;

protected:
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    convective_flux_templated (
        const std::array<real2,nstate> &conservative_soln) const;

    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Get manufactured solution value
    std::array<real,nstate> get_manufactured_solution_value(
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient(
        const dealii::Point<dim,real> &pos) const;

    /// Wall boundary condition
    void boundary_wall (
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Farfield boundary conditions 
    void boundary_farfield (
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif