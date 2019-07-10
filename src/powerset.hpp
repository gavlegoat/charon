/* Greg Anderson
 *
 * Definition of a bounded powerset domain for some underlying domain in the
 * ELINA abstract interpretation library.
 */

#ifndef _POWERSET_H_
#define _POWERSET_H_

#include <elina_abstract0.h>

#include <cstdlib>
#include <vector>
#include <memory>
#include <Eigen/Dense>

/**
 * A thin wrapper class around elina_abstract0_t* which is here to help with
 * memory management.
 */
class Abstract0 {
  public:
    /** The manager associated with this abstract value. */
    elina_manager_t* man;
    /** The wrapped value. */
    elina_abstract0_t* value;
    /** The center of the bounding box of this value. */
    Eigen::VectorXd center;
    /** This bool is true if center is valid. */
    bool center_computed;

    /**
     * Wraps the manager/value pair of m and v into an Abstract0
     *
     * \param m The manager of v.
     * \param v The value to wrap.
     */
    Abstract0(elina_manager_t* m, elina_abstract0_t* v);
    Abstract0(const Abstract0& other);
    Abstract0(Abstract0&& other);

    /**
     * Delete an abstract value. This frees the value but does not free the
     * manager, as it may be shared with other values.
     */
    ~Abstract0();

    Abstract0& operator=(const Abstract0& other);
    Abstract0& operator=(Abstract0&& other);
};

/**
 * Represents a bounded powerset of elements of some underlying abstract
 * domain. For some size N and abstract domain A, a powerset of of size N
 * over A consists of up to N values drawn from A.
 */
class Powerset {
  public:
    /** The number of disjuncts in the domain. */
    int size;
    /** The disjuncts in this powerset. */
    std::vector<std::shared_ptr<Abstract0>> disjuncts;
    /** Copy constructor. */
    Powerset(const Powerset&);
    Powerset(elina_manager_t* man, elina_abstract0_t*, int);
    /** Constructor when we need to compute centers. */
    Powerset(std::vector<std::shared_ptr<Abstract0>>&, int);
    /** Constructor when centers are known. */
    Powerset(std::vector<std::shared_ptr<Abstract0>>&,
        std::vector<Eigen::VectorXd>&, int);

    /**
     * Apply some linear transformation to this powerset.
     *
     * \param dims The dimensions to assign to.
     * \param update The expressions to assign into `dims`
     * \param size The size of `dims` and `update`
     * \param output_dim The dimension of the output abstract value.
     * \return A new powerset representing the modified abstract value.
     */
    Powerset assign_linexpr_array(
        elina_dim_t* dims, elina_linexpr0_t** update,
        unsigned int size,
        unsigned int output_dim) const;

    /**
     * Meet this powerset with some set of linear constraints.
     *
     * \param cons The array of linear constraints to meet with.
     * \return The meet of this value with the constraints.
     */
    Powerset meet_lincons_array(
        elina_lincons0_array_t* cons) const;

    /**
     * Permute the dimensions of this abstract value.
     *
     * \param dp An ELINA dimperm object describing the permutation.
     * \return The permuted abstract value.
     */
    Powerset permute_dimensions(elina_dimperm_t* dp) const;

    /**
     * Remove dimensions from this abstract value.
     *
     * \param dc An ELINA dimchange object describing the dimensions to remove.
     * \return The value with the given dimensions removed.
     */
    Powerset remove_dimensions(elina_dimchange_t* dc) const;

    /**
     * Perform an affine transformation on this abstract value. For each `x` in
     * this abstract value, the result will contain `w*x + b`.
     *
     * \param w The weight matrix of the affine transformation.
     * \param b The bias vector of the affine transformation.
     * \return The transformed abstract value.
     */
    Powerset affine(const Eigen::MatrixXd& w, const Eigen::VectorXd& b) const;

    /**
     * Perform a ReLU on this abstract value. For each dimension i, we split
     * this abstract value around the axis x_i = 0. We assign x_i = 0 in the
     * half where x_i <= 0, then join it with the x_i >= 0 half.
     *
     * \return The abstract value transformed by a ReLU.
     */
    Powerset relu() const;

    /**
     * Join this powerset with another powerset. The size bound of the new
     * powerset will be the greater of the size bounds of this powerset and
     * the other powerset. If the total number of disjuncts in these two
     * powersets is smaller than this bound then the new powerset is
     * initialized with all the disjuncts from both powersets. Otherwise,
     * disjuncts are heuristically chosen and joined until the total number
     * is within the bound.
     *
     * \param other The powerset to join this one with.
     * \return The join of this value and `other`.
     */
    Powerset join(const Powerset& other) const;

    /**
     * Determine whether this powerset's concretization is empty.
     *
     * \return `true` if this abstract value represents bottom.
     */
    bool is_bottom() const;

    /**
     * Get the number of dimensions represented by this value.
     *
     * \return The number of dimensions in the constrained space.
     */
    ssize_t dims() const;

    /**
     * Get a bounding box around this abstract value.
     *
     * \return An ELINA hyperinterval surrounding this abstract value.
     */
    elina_interval_t** bounding_box() const;

    /**
     * Print a bounding box for this powerset.
     */
    void print_bounding_box() const;

    Powerset& operator=(const Powerset& other);
};

#endif
