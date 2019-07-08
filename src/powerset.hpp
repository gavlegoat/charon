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

class Abstract0 {
  public:
    elina_manager_t* man;
    elina_abstract0_t* value;
    Eigen::VectorXd center;
    bool center_computed;

    Abstract0(elina_manager_t*, elina_abstract0_t*);
    ~Abstract0();
    Abstract0& operator=(const Abstract0& other);
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
    /** An ELINA manager which determines the underlying domain A. */
    //elina_manager_t* man;
    /** The disjuncts in this powerset. */
    std::vector<std::shared_ptr<Abstract0>> disjuncts;
    //std::vector<elina_abstract0_t*> disjuncts;
    /** The centers of the bounding boxes of each disjunct. These are used
        to heuristically choose elements to join when necessary. */
    //std::vector<Eigen::VectorXd> centers;
    /** Copy constructor. */
    Powerset(const Powerset&);
    Powerset(elina_manager_t* man, elina_abstract0_t*, int);
    /** Constructor when we need to compute centers. */
    Powerset(std::vector<std::shared_ptr<Abstract0>>&, int);
    //Powerset(std::vector<elina_abstract0_t*>&, int s, elina_manager_t* man);
    /** Constructor when centers are known. */
    Powerset(std::vector<std::shared_ptr<Abstract0>>&,
        std::vector<Eigen::VectorXd>&, int);
    //Powerset(std::vector<elina_abstract0_t*>&, std::vector<Eigen::VectorXd>&,
    //    int, elina_manager_t*);
    /** Destructor. */
    //~Powerset();

    /**
     * Apply some linear transformation to this powerset.
     */
    Powerset assign_linexpr_array(
        elina_dim_t* dims, elina_linexpr0_t** update,
        unsigned int size,
        unsigned int output_dim) const;

    /**
     * Meet this powerset with some set of linear constraints.
     */
    Powerset meet_lincons_array(
        elina_lincons0_array_t* cons) const;

    Powerset permute_dimensions(elina_dimperm_t*) const;
    Powerset remove_dimensions(elina_dimchange_t*) const;

    Powerset affine(const Eigen::MatrixXd& w, const Eigen::VectorXd& b) const;

    Powerset relu() const;

    /**
     * Join this powerset with another powerset. The size bound of the new
     * powerset will be the greater of the size bounds of this powerset and
     * the other powerset. If the total number of disjuncts in these two
     * powersets is smaller than this bound then the new powerset is
     * initialized with all the disjuncts from both powersets. Otherwise,
     * disjuncts are heuristically chosen and joined until the total number
     * is within the bound.
     */
    Powerset join(const Powerset& other) const;

    /**
     * Determine whether this powerset's concretization is empty.
     */
    bool is_bottom() const;

    ssize_t dims() const;

    elina_interval_t** bounding_box() const;
    void print_bounding_box() const;

    /**
     * Copy operator.
     */
    Powerset& operator=(const Powerset& other);
};

#endif
