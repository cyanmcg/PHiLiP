#ifndef __INITIAL_CONDITION_FUNCTION_H__
#define __INITIAL_CONDITION_FUNCTION_H__

// for the initial condition function:
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"
#include "ADTypes.hpp"
#include "../euler.h" // for FreeStreamInitialConditions
#include "../negative_spalart_allmaras_rans_model.h" // for FreeStreamInitialConditions_RANS_SA_negative

namespace PHiLiP {

/// Initial condition function used to initialize a particular flow setup/case
template <int dim, int nstate, typename real>
class InitialConditionFunction : public dealii::Function<dim,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor
    InitialConditionFunction();
    /// Destructor
    ~InitialConditionFunction() {};

    /// Value of the initial condition
    virtual real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const = 0;
};

/// Function used to evaluate farfield conservative solution
template <int dim, int nstate, typename real>
class FreeStreamInitialConditions : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Farfield conservative solution
    std::array<double,nstate> farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store farfield_conservative solution
     */
    FreeStreamInitialConditions (const Physics::Euler<dim,nstate,double> euler_physics)
            : InitialConditionFunction<dim,nstate,real>()
    {
        //const double density_bc = 2.33333*euler_physics.density_inf;
        const double density_bc = euler_physics.density_inf;
        const double pressure_bc = 1.0/(euler_physics.gam*euler_physics.mach_inf_sqr);
        std::array<double,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = density_bc;
        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = euler_physics.velocities_inf[d]; }
        primitive_boundary_values[nstate-1] = pressure_bc;
        farfield_conservative = euler_physics.convert_primitive_to_conservative(primitive_boundary_values);
    }

    /// Returns the istate-th farfield conservative value
    double value (const dealii::Point<dim> &/*point*/, const unsigned int istate) const
    {
        return farfield_conservative[istate];
    }
};

/// Function used to evaluate farfield conservative solution for Reynolds-averaged Navier-Stokes (RANS) with SA negative turbulence model
template <int dim, int nstate, typename real>
class FreeStreamInitialConditions_RANS_SA_negative : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Farfield RANS with SA negative model conservative solution
    std::array<double,nstate> RANS_SA_neg_farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store RANS_SA_neg_farfield_conservative solution
     */
    FreeStreamInitialConditions_RANS_SA_negative (const Physics::NavierStokes<dim,dim+2,double> &rans_physics)
            : InitialConditionFunction<dim,nstate,real>()
    {
        std::array<double,dim+2> RANS_farfield_conservative;

        const double density_ic = rans_physics.density_inf;
        const double pressure_ic = 1.0/(rans_physics.gam*rans_physics.mach_inf_sqr);
        std::array<double,dim+2> primitive_rans_initial_values;
        primitive_rans_initial_values[0] = density_ic;
        for (int d=0;d<dim;d++) { primitive_rans_initial_values[1+d] = rans_physics.velocities_inf[d]; }
        primitive_rans_initial_values[1+dim] = pressure_ic;
        RANS_farfield_conservative = rans_physics.convert_primitive_to_conservative(primitive_rans_initial_values);

        const double sa_viscosity_ic = 3.0*rans_physics.viscosity_coefficient_inf;
        const double SA_neg_farfield_conservative = density_ic*sa_viscosity_ic;
        for (int i=0;i<dim+2;++i){
            RANS_SA_neg_farfield_conservative[i] = RANS_farfield_conservative[i];
        }
        for (int i=dim+2;i<nstate;++i){
            RANS_SA_neg_farfield_conservative[i] = SA_neg_farfield_conservative;
        }
    }

    /// Returns the istate-th farfield conservative value
    double value (const dealii::Point<dim> &/*point*/, const unsigned int istate) const
    {
        return RANS_SA_neg_farfield_conservative[istate];
    }
};

/// Initial Condition Function: Taylor Green Vortex (uniform density)
template <int dim, int nstate, typename real>
class InitialConditionFunction_TaylorGreenVortex : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for TaylorGreenVortex_InitialCondition with uniform density
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  Reference: (1) Gassner2016split, 
     *             (2) de la Llave Plata et al. (2019). "On the performance of a high-order multiscale DG approach to LES at increasing Reynolds number."
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    InitialConditionFunction_TaylorGreenVortex (
        const double       gamma_gas = 1.4,
        const double       mach_inf = 0.1);

    const double gamma_gas; ///< Constant heat capacity ratio of fluid.
    const double mach_inf; ///< Farfield Mach number.
    const double mach_inf_sqr; ///< Farfield Mach number squared.
        
    /// Value of initial condition expressed in terms of conservative variables
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;

protected:
    /// Value of initial condition expressed in terms of primitive variables
    real primitive_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;
    
    /// Converts value from: primitive to conservative
    real convert_primitive_to_conversative_value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const;

    /// Value of initial condition for density
    virtual real density(const dealii::Point<dim,real> &point) const;
};

/// Initial Condition Function: Taylor Green Vortex (isothermal density)
template <int dim, int nstate, typename real>
class InitialConditionFunction_TaylorGreenVortex_Isothermal
    : public InitialConditionFunction_TaylorGreenVortex<dim,nstate,real>
{
public:
    /// Constructor for TaylorGreenVortex_InitialCondition with isothermal density
    /** Calls the Function(const unsigned int n_components) constructor in deal.II
     *  This sets the public attribute n_components = nstate, which can then be accessed
     *  by all the other functions
     *  -- Reference: (1) Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                    between the spectral difference and flux reconstruction schemes." 
     *                    Computers & Fluids 221 (2021): 104922.
     *                (2) Brian Vermeire 2014 Thesis  
     *  These initial conditions are given in nondimensional form (free-stream as reference)
     */
    InitialConditionFunction_TaylorGreenVortex_Isothermal (
        const double       gamma_gas = 1.4,
        const double       mach_inf = 0.1);

protected:
    /// Value of initial condition for density
    real density(const dealii::Point<dim,real> &point) const override;
};

/// Initial Condition Function: 1D Burgers Rewienski
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersRewienski: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersRewienski ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Viscous
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersViscous: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersRewienski
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersViscous ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersInviscid: public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersInviscid
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersInviscid ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_BurgersInviscidEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_BurgersInviscidEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_BurgersInviscidEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: 1D Burgers Inviscid
template <int dim, int nstate, typename real>
class InitialConditionFunction_Advection
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_Inviscid
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_Advection ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Advection Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_AdvectionEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_AdvectionEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_AdvectionEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Convection Diffusion Orders of Accuracy
template <int dim, int nstate, typename real>
class InitialConditionFunction_ConvDiff
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_ConvDiffEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_ConvDiff ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: Convection Diffusion Energy
template <int dim, int nstate, typename real>
class InitialConditionFunction_ConvDiffEnergy
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_ConvDiffEnergy
    /** Calls the Function(const unsigned int n_components) constructor in deal.II*/
    InitialConditionFunction_ConvDiffEnergy ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate) const override;
};

/// Initial Condition Function: 1D Sine Function; used for temporal convergence
template <int dim, int nstate, typename real>
class InitialConditionFunction_1DSine
        : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on

public:
    /// Constructor for InitialConditionFunction_1DSine
    InitialConditionFunction_1DSine ();

    /// Value of initial condition
    real value (const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial condition 0.
template <int dim, int nstate, typename real>
class InitialConditionFunction_Zero : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on
    
public:
    /// Constructor
    InitialConditionFunction_Zero();

    /// Returns zero.
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial condition positive constant.
template <int dim, int nstate, typename real>
class InitialConditionFunction_PositiveConstant : public InitialConditionFunction<dim,nstate,real>
{
protected:
    using dealii::Function<dim,real>::value; ///< dealii::Function we are templating on
    
public:
    /// Constructor
    InitialConditionFunction_PositiveConstant();

    /// Returns a positive constant value.
    real value(const dealii::Point<dim,real> &point, const unsigned int istate = 0) const override;
};

/// Initial condition function factory
template <int dim, int nstate, typename real>
class InitialConditionFactory
{
protected:    
    /// Enumeration of all flow solver initial conditions types defined in the Parameters class
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    /// Enumeration of all taylor green vortex initial condition sub-types defined in the Parameters class
    using DensityInitialConditionEnum = Parameters::FlowSolverParam::DensityInitialConditionType;

public:
    /// Construct InitialConditionFunction object from global parameter file
    static std::shared_ptr<InitialConditionFunction<dim,nstate,real>>
    create_InitialConditionFunction(
        Parameters::AllParameters const *const param);
};

} // PHiLiP namespace
#endif
