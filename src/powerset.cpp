#include <zonotope.h>
#include <elina_box_internal.h>

#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <elina_box_internal.h>

#include "powerset.hpp"

Abstract0::Abstract0(elina_manager_t* m, elina_abstract0_t* a) {
  man = m;
  value = a;
  center_computed = false;
}

Abstract0& Abstract0::operator=(const Abstract0& other) {
  man = other.man;
  value = other.value;
  center_computed = false;
  return *this;
}

Abstract0::~Abstract0() {
  elina_abstract0_free(man, value);
}

Eigen::VectorXd compute_center(elina_manager_t* man, elina_abstract0_t* abs) {
  elina_interval_t** itv;
  const char* domain = man->library;
  bool is_zonotope = (std::strcmp(domain, "Zonotope") == 0);
  bool is_interval = (std::strcmp(domain, "elina box") == 0);
  if (!is_zonotope && !is_interval) {
    itv = elina_abstract0_to_box(man, abs);
  }
  int dims = elina_abstract0_dimension(man, abs).realdim;
  Eigen::VectorXd center(dims);
  for (int i = 0; i < dims; i++) {
    double l, u;
    if (is_zonotope) {
      elina_double_set_scalar(&l, ((zonotope_t*) abs->value)->box[i]->inf, MPFR_RNDN);
      elina_double_set_scalar(&u, ((zonotope_t*) abs->value)->box[i]->sup, MPFR_RNDN);
    } else if (is_interval) {
      elina_double_set_scalar(&l, ((elina_box_t*) abs->value)->p[i]->inf, MPFR_RNDN);
      elina_double_set_scalar(&u, ((elina_box_t*) abs->value)->p[i]->sup, MPFR_RNDN);
    } else {
      elina_double_set_scalar(&l, itv[i]->inf, MPFR_RNDN);
      elina_double_set_scalar(&u, itv[i]->sup, MPFR_RNDN);
    }
    center(i) = (l + u) / 2.0;
  }

  //for (int i = 0; i < dims; i++) {
  //  elina_interval_free(itv[i]);
  //}
  //free(itv);

	return center;
}

Powerset::Powerset(const Powerset& other) {
  //man = other.man;

  //disjuncts = std::vector<elina_abstract0_t*>();
  //for (elina_abstract0_t* it : other.disjuncts) {
  //  disjuncts.push_back(elina_abstract0_copy(man, it));
  //}
  //disjuncts = std::vector<std::shared_ptr<Abstract0>>(other.disjuncts);
  disjuncts = other.disjuncts;

  size = other.size;
  //centers = std::vector<Eigen::VectorXd>(other.centers);
}

Powerset::Powerset(elina_manager_t* m, elina_abstract0_t* a, int s) {
  size = s;
  disjuncts = std::vector<std::shared_ptr<Abstract0>>();
  disjuncts.push_back(std::shared_ptr<Abstract0>(new Abstract0(m, a)));
}

//Powerset::Powerset(std::vector<elina_abstract0_t*>& ds, int s, elina_manager_t* m) {
Powerset::Powerset(std::vector<std::shared_ptr<Abstract0>>& ds, int s) {
  size = s;
  //man = m;

  //disjuncts = std::vector<elina_abstract0_t*>();
  //for (elina_abstract0_t* it : ds) {
  //  disjuncts.push_back(elina_abstract0_copy(man, it));
  //}propagate_through_network

  //disjuncts = std::vector<std::shared_ptr<Abstract0>>(ds);
  disjuncts = ds;
  //if (ds.empty()) {
  //  man = NULL;
  //} else {
  //  man = ds[0]->man;
  //}

  //centers = std::vector<Eigen::VectorXd>();
  //for (elina_abstract0_t* it : disjuncts) {
  //  centers.push_back(compute_center(man, it));
  //}
  //for (std::shared_ptr<Abstract0> it : disjuncts) {
  //  centers.push_back(compute_center(it->man, it->value));
  //}
}

//Powerset::Powerset(std::vector<elina_abstract0_t*>& ds,
//    std::vector<Eigen::VectorXd>& cs, int s, elina_manager_t* m) {
Powerset::Powerset(std::vector<std::shared_ptr<Abstract0>>& ds,
    std::vector<Eigen::VectorXd>& cs, int s) {
  size = s;
  //man = m;

  //disjuncts = std::vector<elina_abstract0_t*>();
  //for (elina_abstract0_t* it : ds) {
  //  disjuncts.push_back(elina_abstract0_copy(man, it));
  //}
  disjuncts = std::vector<std::shared_ptr<Abstract0>>(ds);
  //if (ds.empty()) {
  //  man = NULL;
  //} else {
  //  man = ds[0]->man;
  //}

  //centers = std::vector<Eigen::VectorXd>(cs);
}

//Powerset::~Powerset() {
//  for (elina_abstract0_t* it : disjuncts) {
//    elina_abstract0_free(man, it);
//  }
//}

Powerset Powerset::assign_linexpr_array(elina_dim_t* dims,
    elina_linexpr0_t** update, unsigned int size, unsigned int s) const {
  //std::vector<elina_abstract0_t*> ds;
  std::vector<std::shared_ptr<Abstract0>> ds;
  //for (elina_abstract0_t* it : this->disjuncts) {
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* it_dim;
    size_t num_dims = elina_abstract0_dimension(it->man, it->value).realdim;
    if (s > num_dims) {
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
      elina_abstract0_free(it->man, abs);
      continue;
    }
    if (num_dims > s) {
      elina_dimchange_t* dc = elina_dimchange_alloc(0, num_dims - s);
      for (unsigned int i = 0; i < num_dims - s; i++) {
        dc->dim[i] = s + i;
      }
      elina_abstract0_t* abs2 = elina_abstract0_remove_dimensions(it->man, false, abs, dc);
      elina_abstract0_free(it->man, abs);
      abs = abs2;
      elina_dimchange_free(dc);
    }
    // If the new disjunct is bottom there's no need to add it to the powerset,
    // so we'll drop it for efficiency.
    //ds.push_back(abs);
    ds.push_back(std::shared_ptr<Abstract0>(new Abstract0(it->man, abs)));
  }
  //Powerset p(ds, this->size, this->man);
  Powerset p(ds, this->size);
  //for (elina_abstract0_t* it : ds) {
  //  elina_abstract0_free(this->man, it);
  //}
  return p;
}

Powerset Powerset::meet_lincons_array(elina_lincons0_array_t* cons) const {
  //std::vector<elina_abstract0_t*> ds;
  std::vector<std::shared_ptr<Abstract0>> ds;
  //for (elina_abstract0_t* it : this->disjuncts) {
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_meet_lincons_array(
        it->man, false, it->value, cons);
    bool bot = elina_abstract0_is_bottom(it->man, abs);
    if (!bot) {
      //ds.push_back(abs);
      ds.push_back(std::shared_ptr<Abstract0>(new Abstract0(it->man, abs)));
    } else {
      elina_abstract0_free(it->man, abs);
    }
  }
  //Powerset p(ds, this->size, this->man);
  Powerset p(ds, this->size);
  //for (elina_abstract0_t* it : ds) {
  //  elina_abstract0_free(this->man, it);
  //}
  return p;
}

Powerset Powerset::permute_dimensions(elina_dimperm_t* dp) const {
  //std::vector<elina_abstract0_t*> ds;
  std::vector<std::shared_ptr<Abstract0>> ds;
  //for (elina_abstract0_t* it : this->disjuncts) {
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_permute_dimensions(
        it->man, false, it->value, dp);
    ds.push_back(std::shared_ptr<Abstract0>(new Abstract0(it->man, abs)));
  }
  //Powerset p(ds, this->size, this->man);
  Powerset p(ds, this->size);
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
  //for (elina_abstract0_t* it : ds) {
  //  elina_abstract0_free(this->man, it);
  //}
  return p;
}

Powerset Powerset::remove_dimensions(elina_dimchange_t* dc) const {
  //std::vector<elina_abstract0_t*> ds;
  std::vector<std::shared_ptr<Abstract0>> ds;
  //for (elina_abstract0_t* it : this->disjuncts) {
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    elina_abstract0_t* abs = elina_abstract0_remove_dimensions(
        it->man, false, it->value, dc);
    bool bot = elina_abstract0_is_bottom(it->man, abs);
    if (!bot) {
      //ds.push_back(abs);
      ds.push_back(std::shared_ptr<Abstract0>(new Abstract0(it->man, abs)));
    } else {
      elina_abstract0_free(it->man, abs);
    }
  }
  //Powerset p(ds, this->size, this->man);
  Powerset p(ds, this->size);
  if (disjuncts.size() > 0) {
    unsigned int dims = elina_abstract0_dimension(disjuncts[0]->man, disjuncts[0]->value).realdim;
    unsigned int out_dims = dims - dc->realdim;
    for (unsigned int i = 0; i < disjuncts.size(); i++) {
      if (disjuncts[i]->center_computed) {
        p.disjuncts[i]->center_computed = true;
        Eigen::VectorXd c(out_dims);
        int dc_ind = 0;
        int c_ind = 0;
        for (unsigned int j = 0; j < dims; j++) {
          if (dc->dim[dc_ind] == j) {
            continue;
          }
          c(c_ind) = disjuncts[i]->center(j);
        }
      }
    }
  }
  //for (elina_abstract0_t* it : ds) {
  //  elina_abstract0_free(this->man, it);
  //}
  return p;
}

Powerset Powerset::affine(const Eigen::MatrixXd& m, const Eigen::VectorXd& b) const {
  int in_size = m.cols();
  int out_size = m.rows();
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
  //size_t num_dims = elina_abstract0_dimension(man, disjuncts[0]).realdim;
  size_t num_dims = elina_abstract0_dimension(disjuncts[0]->man, disjuncts[0]->value).realdim;
  for (unsigned int i = 0; i < num_dims; i++) {
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

    Powerset zlt = z.meet_lincons_array(&lt0);
    z = z.meet_lincons_array(&gt0);

    elina_linexpr0_t* zero = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 0);
    elina_linexpr0_set_cst_scalar_double(zero, 0.0);
    elina_dim_t dim = i;
    zlt = zlt.assign_linexpr_array(&dim, &zero, 1, num_dims);

    z = z.join(zlt);

    elina_linexpr0_free(zero);
    elina_lincons0_array_clear(&lt0);
    elina_lincons0_array_clear(&gt0);
  }

  return z;
}

// NOTE: by convention, the this and other should point to the same manager.
// The manager of this is used, so if it is not compatible with the manager of
// other I'm not sure what happens.
Powerset Powerset::join(const Powerset& other) const {
  if (this->disjuncts.size() == 0) {
    return other;
  } else if (other.disjuncts.size() == 0) {
    return *this;
  }
  //std::vector<elina_abstract0_t*> ds;
  //for (elina_abstract0_t* it : this->disjuncts) {
  //  ds.push_back(elina_abstract0_copy(man, it));
  //}
  //for (elina_abstract0_t* it : other.disjuncts) {
  //  ds.push_back(elina_abstract0_copy(man, it));
  //}
  std::vector<std::shared_ptr<Abstract0>> ds(disjuncts);
  ds.insert(ds.end(), other.disjuncts.begin(), other.disjuncts.end());
  //std::vector<Eigen::VectorXd> cs(centers);
  //cs.insert(cs.end(), other.centers.begin(), other.centers.end());
  // Now all holds all of the disjuncts from both vectors, so we need to join
  // individual elements until the total number of disjuncts is amll enough
  unsigned int s = std::max(size, other.size);
  while (ds.size() > s) {
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
    elina_abstract0_t* n = elina_abstract0_join(ds[best_i]->man, false,
        ds[best_i]->value, ds[best_j]->value);
    Abstract0* newAbs0 = new Abstract0(ds[best_i]->man, n);
    //elina_abstract0_free(man, ds[best_i]);
    //elina_abstract0_free(man, ds[best_j]);
    // j > i so we don't need to worry about messing up indices if we erase
    // j first
    ds.erase(ds.begin() + best_j);
    ds.erase(ds.begin() + best_i);
    //cs.erase(cs.begin() + best_j);
    //cs.erase(cs.begin() + best_i);
    ds.push_back(std::shared_ptr<Abstract0>(newAbs0));
    //cs.push_back(compute_center(ds[best_i]->man, n));
  }

  //Powerset p(ds, cs, s, man);
  //Powerset p(ds, cs, s);
  Powerset p(ds, s);
  //for (elina_abstract0_t* it : ds) {
  //  elina_abstract0_free(man, it);
  //}
  return p;
}

bool Powerset::is_bottom() const {
  bool bottom = true;
  //for (elina_abstract0_t* it : this->disjuncts) {
  for (std::shared_ptr<Abstract0> it : this->disjuncts) {
    if (!elina_abstract0_is_bottom(it->man, it->value)) {
      bottom = false;
      break;
    }
  }
  // Note that if disjuncts is empty then this powerset is bottom.

  return bottom;
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
  //this->man = other.man;

  // If we're going to change this powerset, we need to free our disjuncts
  // first
  //for (elina_abstract0_t* it : this->disjuncts) {
  //  //if (other.disjuncts.size() >= 1) {
  //  //  std::cout << "this: " << ((zonotope_t*) it->value)->paf[0] << " other: "
  //  //    << ((zonotope_t*) other.disjuncts[0]->value)->paf[0] << " pby: "
  //  //    << ((zonotope_t*) it->value)->paf[0]->pby << std::endl;
  //  //}
  //  elina_abstract0_free(this->man, it);
  //}

  //this->disjuncts = std::vector<elina_abstract0_t*>();
  //for (elina_abstract0_t* it : other.disjuncts) {
  //  elina_abstract0_t* n = elina_abstract0_copy(man, it);
  //  this->disjuncts.push_back(n);
  //}
  disjuncts = other.disjuncts;

  //centers = std::vector<Eigen::VectorXd>(other.centers);
  return *this;
}

ssize_t Powerset::dims() const {
  if (this->disjuncts.size() > 0)
    return elina_abstract0_dimension(disjuncts[0]->man, this->disjuncts[0]->value).realdim;
  return 0;
}

