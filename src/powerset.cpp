#include <zonotope.h>
#include <elina_box_internal.h>

#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <elina_box_internal.h>

#include "powerset.hpp"

Abstract0::Abstract0(elina_manager_t* m, elina_abstract0_t* a):
  man{m}, value{a}, center_computed{false} {}

Abstract0::Abstract0(const Abstract0& other):
  man{other.man}, value{elina_abstract0_copy(other.man, other.value)},
  center{other.center}, center_computed{other.center_computed} {}

Abstract0::Abstract0(Abstract0&& other):
  man{other.man}, value{other.value}, center{other.center},
  center_computed{other.center_computed} {
  other.value = nullptr;
}

Abstract0::~Abstract0() {
  if (value != nullptr) {
    elina_abstract0_free(man, value);
  }
}

Abstract0& Abstract0::operator=(const Abstract0& other) {
  this->man = other.man;
  this->value = elina_abstract0_copy(other.man, other.value);
  this->center = other.center;
  this->center_computed = other.center_computed;
  return *this;
}

Abstract0& Abstract0::operator=(Abstract0&& other) {
  this->man = other.man;
  this->value = other.value;
  other.value = nullptr;
  this->center = other.center;
  this->center_computed = other.center_computed;
  return *this;
}

Eigen::VectorXd compute_center(elina_manager_t* man, elina_abstract0_t* abs) {
  elina_interval_t** itv;
  itv = elina_abstract0_to_box(man, abs);
  int dims = elina_abstract0_dimension(man, abs).realdim;
  Eigen::VectorXd center(dims);
  for (int i = 0; i < dims; i++) {
    double l, u;
    elina_double_set_scalar(&l, itv[i]->inf, MPFR_RNDN);
    elina_double_set_scalar(&u, itv[i]->sup, MPFR_RNDN);
    center(i) = (l + u) / 2.0;
  }
  return center;
}

Powerset::Powerset(const Powerset& other) {
  disjuncts = other.disjuncts;
  size = other.size;
}

Powerset::Powerset(elina_manager_t* m, elina_abstract0_t* a, int s) {
  size = s;
  disjuncts = std::vector<std::shared_ptr<Abstract0>>();
  disjuncts.push_back(std::make_shared<Abstract0>(m, a));
}

Powerset::Powerset(std::vector<std::shared_ptr<Abstract0>>& ds, int s) {
  size = s;
  disjuncts = ds;
}

Powerset::Powerset(std::vector<std::shared_ptr<Abstract0>>& ds,
    std::vector<Eigen::VectorXd>& cs, int s) {
  size = s;
  disjuncts = std::vector<std::shared_ptr<Abstract0>>(ds);
}

Powerset Powerset::assign_linexpr_array(elina_dim_t* dims,
    elina_linexpr0_t** update, unsigned int size, unsigned int s) const {
  std::vector<std::shared_ptr<Abstract0>> ds;
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* it_dim;
    size_t num_dims = elina_abstract0_dimension(it->man, it->value).realdim;
    if (s > num_dims) {
      // If the output size is greater than the input size, then we need to
      // add dimensions to the input abstract value.
      elina_dimchange_t* dc = elina_dimchange_alloc(0, s - num_dims);
      for (unsigned int i = 0; i < s - num_dims; i++) {
        dc->dim[i] = num_dims;
      }
      it_dim = elina_abstract0_add_dimensions(it->man, false, it->value, dc, false);
      elina_dimchange_free(dc);
    } else {
      it_dim = elina_abstract0_copy(it->man, it->value);
    }

    elina_abstract0_t* abs = elina_abstract0_assign_linexpr_array(
        it->man, false, it_dim, dims, update, size, NULL);
    elina_abstract0_free(it->man, it_dim);
    bool bot = elina_abstract0_is_bottom(it->man, abs);
    if (bot) {
      // If this value is bottom we don't need to add it back to the powerset.
      elina_abstract0_free(it->man, abs);
      continue;
    }

    if (num_dims > s) {
      // If the input size is greater than the output size, then we need to
      // remove excess dimensions here.
      elina_dimchange_t* dc = elina_dimchange_alloc(0, num_dims - s);
      for (unsigned int i = 0; i < num_dims - s; i++) {
        dc->dim[i] = s + i;
      }
      elina_abstract0_t* abs2 = elina_abstract0_remove_dimensions(it->man, false, abs, dc);
      elina_abstract0_free(it->man, abs);
      abs = abs2;
      elina_dimchange_free(dc);
    }

    ds.push_back(std::make_shared<Abstract0>(it->man, abs));
  }

  return Powerset(ds, this->size);
}

Powerset Powerset::meet_lincons_array(elina_lincons0_array_t* cons) const {
  std::vector<std::shared_ptr<Abstract0>> ds;
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_meet_lincons_array(
        it->man, false, it->value, cons);
    bool bot = elina_abstract0_is_bottom(it->man, abs);
    // If this value is bottom it doesn't affect the powerset.
    if (!bot) {
      ds.push_back(std::make_shared<Abstract0>(it->man, abs));
    } else {
      elina_abstract0_free(it->man, abs);
    }
  }
  return Powerset(ds, this->size);
}

Powerset Powerset::permute_dimensions(elina_dimperm_t* dp) const {
  std::vector<std::shared_ptr<Abstract0>> ds;
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_permute_dimensions(
        it->man, false, it->value, dp);
    ds.push_back(std::make_shared<Abstract0>(it->man, abs));
  }
  Powerset p(ds, this->size);
  // Permute the centers as well so we don't need to recompute them.
  if (disjuncts.size() > 0) {
    int dims = elina_abstract0_dimension(disjuncts[0]->man, disjuncts[0]->value).realdim;
    Eigen::VectorXi perm(dims);
    for (int i = 0; i < dims; i++) {
      perm(i) = dp->dim[i];
    }
    Eigen::PermutationMatrix<Eigen::Dynamic> pm(perm);
    for (unsigned int i = 0; i < disjuncts.size(); i++) {
      if (disjuncts[i]->center_computed) {
        p.disjuncts[i]->center_computed = true;
        p.disjuncts[i]->center = pm * disjuncts[i]->center;
      }
    }
  }
  return p;
}

Powerset Powerset::remove_dimensions(elina_dimchange_t* dc) const {
  std::vector<std::shared_ptr<Abstract0>> ds;
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_remove_dimensions(
        it->man, false, it->value, dc);
    bool bot = elina_abstract0_is_bottom(it->man, abs);
    // bottom elements don't need to be added to this powerset.
    if (!bot) {
      ds.push_back(std::make_shared<Abstract0>(it->man, abs));
    } else {
      elina_abstract0_free(it->man, abs);
    }
  }
  Powerset p(ds, this->size);
  // Update centers
  if (disjuncts.size() > 0) {
    unsigned int dims = elina_abstract0_dimension(
        disjuncts[0]->man, disjuncts[0]->value).realdim;
    unsigned int out_dims = dims - dc->realdim;
    for (unsigned int i = 0; i < disjuncts.size(); i++) {
      if (disjuncts[i]->center_computed) {
        p.disjuncts[i]->center_computed = true;
        Eigen::VectorXd c(out_dims);
        int dc_ind = 0;
        int c_ind = 0;
        for (unsigned int j = 0; j < dims; j++) {
          if (dc->dim[dc_ind] == j) {
            dc_ind++;
            continue;
          }
          c(c_ind) = disjuncts[i]->center(j);
          c_ind++;
        }
      }
    }
  }
  return p;
}

Powerset Powerset::affine(const Eigen::MatrixXd& m, const Eigen::VectorXd& b) const {
  int in_size = m.cols();
  int out_size = m.rows();

  // Create an elina linexpr array representing this update
  elina_dim_t* dims = (elina_dim_t*) malloc(out_size * sizeof(elina_dim_t));
  elina_linexpr0_t** update = (elina_linexpr0_t**) malloc(out_size *
      sizeof(elina_linexpr0_t*));
  for (int j = 0; j < out_size; j++) {
    dims[j] = j;
    update[j] = elina_linexpr0_alloc(ELINA_LINEXPR_DENSE, in_size);
    for (int k = 0; k < in_size; k++) {
      elina_linexpr0_set_coeff_scalar_double(update[j], k, m(j,k));
    }
    elina_linexpr0_set_cst_scalar_double(update[j], b(j));
  }

  Powerset z = assign_linexpr_array(dims, update, out_size, out_size);

  free(dims);
  for (int j = 0; j < out_size; j++) {
    elina_linexpr0_free(update[j]);
  }
  free(update);

  return z;
}

Powerset Powerset::relu() const {
  Powerset z = *this;
  size_t num_dims = elina_abstract0_dimension(
      disjuncts[0]->man, disjuncts[0]->value).realdim;
  for (unsigned int i = 0; i < num_dims; i++) {
    // Create two linear constraints, so that we can meet z with
    // x_i <= 0 and x_i >= 0.
    elina_linexpr0_t* lt0_le = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 1);
    elina_linexpr0_t* gt0_le = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 1);
    elina_linexpr0_set_coeff_scalar_double(lt0_le, i, -1.0);
    elina_linexpr0_set_coeff_scalar_double(gt0_le, i, 1.0);

    elina_lincons0_array_t lt0 = elina_lincons0_array_make(1);
    elina_lincons0_array_t gt0 = elina_lincons0_array_make(1);
    lt0.p[0].constyp = ELINA_CONS_SUPEQ;
    lt0.p[0].linexpr0 = lt0_le;
    gt0.p[0].constyp = ELINA_CONS_SUP;
    gt0.p[0].linexpr0 = gt0_le;

    // zlt = z `meet` x_i <= 0
    Powerset zlt = z.meet_lincons_array(&lt0);
    // z = z `meet` x_i >= 0
    z = z.meet_lincons_array(&gt0);

    // Assign x_i = 0 in zlt
    elina_linexpr0_t* zero = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 0);
    elina_linexpr0_set_cst_scalar_double(zero, 0.0);
    elina_dim_t dim = i;
    zlt = zlt.assign_linexpr_array(&dim, &zero, 1, num_dims);

    // Join z with the modified zlt
    z = z.join(zlt);

    elina_linexpr0_free(zero);
    elina_lincons0_array_clear(&lt0);
    elina_lincons0_array_clear(&gt0);
  }

  return z;
}

Powerset Powerset::join(const Powerset& other) const {
  if (this->disjuncts.size() == 0) {
    return other;
  } else if (other.disjuncts.size() == 0) {
    return *this;
  }
  std::vector<std::shared_ptr<Abstract0>> ds(disjuncts);
  ds.insert(ds.end(), other.disjuncts.begin(), other.disjuncts.end());
  // ds now holds all of the disjuncts from both powersets
  unsigned int s = std::max(size, other.size);
  while (ds.size() > s) {
    // Find the two disjuncts whose centers are closes to each other
    int best_i = 0, best_j = 1;
    if (s > 1) {
      if (!ds[0]->center_computed) {
        ds[0]->center = compute_center(ds[0]->man, ds[0]->value);
      }
      if (!ds[1]->center_computed) {
        ds[1]->center = compute_center(ds[1]->man, ds[1]->value);
      }
      double best_dist = (ds[0]->center - ds[1]->center).norm();
      for (unsigned int i = 0; i < ds.size(); i++) {
        if (!ds[i]->center_computed) {
          ds[i]->center = compute_center(ds[i]->man, ds[i]->value);
        }
        for (unsigned int j = i+1; j < ds.size(); j++) {
          if (!ds[j]->center_computed) {
            ds[j]->center = compute_center(ds[j]->man, ds[j]->value);
          }
          double dist = (ds[i]->center - ds[j]->center).norm();
          if (dist < best_dist) {
            best_i = i;
            best_j = j;
            best_dist = dist;
          }
        }
      }
    }
    // Join those two disjuncts
    elina_abstract0_t* n = elina_abstract0_join(ds[best_i]->man, false,
        ds[best_i]->value, ds[best_j]->value);
    // j > i so we don't need to worry about messing up indices if we erase
    // j first
    ds.erase(ds.begin() + best_j);
    ds.erase(ds.begin() + best_i);
    ds.push_back(std::make_shared<Abstract0>(ds[best_i]->man, n));
  }

  return Powerset(ds, s);
}

bool Powerset::is_bottom() const {
  bool bottom = true;
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    if (!elina_abstract0_is_bottom(it->man, it->value)) {
      bottom = false;
      break;
    }
  }
  // Note that if disjuncts is empty then this powerset is bottom.

  return bottom;
}

ssize_t Powerset::dims() const {
  if (this->disjuncts.size() > 0)
    return elina_abstract0_dimension(
        this->disjuncts[0]->man, this->disjuncts[0]->value).realdim;
  return 0;
}

elina_interval_t** Powerset::bounding_box() const {
  elina_interval_t** itvs[disjuncts.size()];
  for (unsigned int i = 0; i < disjuncts.size(); i++) {
    itvs[i] = elina_abstract0_to_box(disjuncts[i]->man, disjuncts[i]->value);
  }
  size_t dims = elina_abstract0_dimension(disjuncts[0]->man, disjuncts[0]->value).realdim;
  elina_interval_t** ret = (elina_interval_t**) malloc(dims * sizeof(elina_interval_t*));
  for (unsigned int i = 0; i < dims; i++) {
    ret[i] = elina_interval_alloc();
    double lowest = std::numeric_limits<double>::max();
    double highest = -std::numeric_limits<double>::max();
    for (unsigned int j = 0; j < disjuncts.size(); j++) {
      double l, u;
      elina_double_set_scalar(&l, itvs[j][i]->inf, MPFR_RNDN);
      elina_double_set_scalar(&u, itvs[j][i]->sup, MPFR_RNDN);
      if (l < lowest) {
        lowest = l;
      }
      if (u > highest) {
        highest = u;
      }
    }
    elina_interval_set_double(ret[i], lowest, highest);
  }
  for (unsigned int i = 0; i < disjuncts.size(); i++) {
    for (unsigned int j = 0; j < dims; j++) {
      elina_interval_free(itvs[i][j]);
    }
    free(itvs[i]);
  }
  return ret;
}

void Powerset::print_bounding_box() const {
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  elina_interval_t** box = bounding_box();
  for (int i = 0; i < dims(); i++) {
    double l, u;
    elina_double_set_scalar(&l, box[i]->inf, MPFR_RNDN);
    elina_double_set_scalar(&u, box[i]->sup, MPFR_RNDN);
    std::cout << "[" << l << ", " << u << "]" << std::endl;
    elina_interval_free(box[i]);
  }
  free(box);
}

Powerset& Powerset::operator=(const Powerset& other) {
  this->size = other.size;
  disjuncts = other.disjuncts;
  return *this;
}

