/* Greg Anderson
 * Classes and utilities for representing neural networks.
 */

#include <zonotope.h>

#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <elina_scalar.h>

#include "network.hpp"
#include "powerset.hpp"

// Element-wise ReLU over a vector
Eigen::VectorXd relu(const Eigen::VectorXd& x) {
  Eigen::VectorXd y = x;
  for (unsigned int i = 0; i < y.size(); i++) {
    if (y(i) < 0.0) {
      y(i) = 0.0;
    }
  }
  return y;
}

// Element-wise ReLU over a 3D tensor
std::vector<Eigen::MatrixXd> relu(const std::vector<Eigen::MatrixXd>& x) {
  std::vector<Eigen::MatrixXd> y;
  for (Eigen::MatrixXd m : x) {
    Eigen::MatrixXd o(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++) {
      for (int j = 0; j < m.cols(); j++) {
        o(i,j) = std::max(m(i,j), 0.0);
      }
    }
    y.push_back(o);
  }
  return y;
}

Layer::Layer(int iw, int ih, int id, int ow, int oh, int od) :
  input_width{iw}, input_height{ih}, input_depth{id}, output_width{ow},
  output_height{oh}, output_depth{od} {}

MaxPoolLayer::MaxPoolLayer(int ww, int wh, int iw, int ih, int d) :
    Layer(iw, ih, d, iw / ww, ih / wh, d), window_width(ww), window_height(wh) {
  if (iw % ww != 0 || ih % wh != 0) {
    throw std::runtime_error("Bad initialization of MaxPoolLayer");
  }
}

LayerType MaxPoolLayer::get_type() const {
  return MP;
}

std::vector<Eigen::MatrixXd> MaxPoolLayer::evaluate(
    const std::vector<Eigen::MatrixXd>& x) const {
  std::vector<Eigen::MatrixXd> ret;
  for (int k = 0; k < input_depth; k++) {
    ret.push_back(Eigen::MatrixXd(input_height / window_height,
          input_width / window_width));
    for (int i = 0; i < input_height / window_height; i++) {
      for (int j = 0; j < input_width / window_width; j++) {
        double max = -std::numeric_limits<double>::max();
        for (int i2 = 0; i2 < window_height; i2++) {
          for (int j2 = 0; j2 < window_width; j2++) {
            if (x[k](window_height*i + i2, window_width*j + j2) > max) {
              max = x[k](window_height*i + i2, window_width*j + j2);
            }
          }
        }
        ret[k](i,j) = max;
      }
    }
  }
  return ret;
}

std::vector<Eigen::MatrixXd> MaxPoolLayer::backpropagate(
    const std::vector<Eigen::MatrixXd>& eval,
    const std::vector<Eigen::MatrixXd>& grad) const {
  std::vector<Eigen::MatrixXd> ret;
  for (int i = 0; i < input_depth; i++) {
    ret.push_back(Eigen::MatrixXd(input_height, input_width));
    for (int j = 0; j < output_height; j++) {
      for (int k = 0; k < output_width; k++) {
        int max_j2 = 0;
        int max_k2 = 0;
        double max = eval[i](window_height * j, window_width * k);
        for (int j2 = 0; j2 < window_height; j2++) {
          for (int k2 = 0; k2 < window_width; k2++) {
            if (eval[i](window_height*j + j2, window_width*k + k2) > max) {
              max = eval[i](window_height*j + j2, window_width*k + k2);
              max_j2 = j2;
              max_k2 = k2;
            }
          }
        }
        for (int j2 = 0; j2 < window_height; j2++) {
          for (int k2 = 0; k2 < window_width; k2++) {
            if (j2 == max_j2 && k2 == max_k2) {
              ret[i](window_height*j + j2, window_width*k + k2) = grad[i](j,k);
            } else {
              ret[i](window_height*j + j2, window_width*k + k2) = 0.0;
            }
          }
        }
      }
    }
  }
  return ret;
}

Powerset MaxPoolLayer::propagate_powerset(const Powerset& pow) const {
  // This is taken from the appendix of the AI2 paper
  // Rearrange the elements of p into blocks
  int dim = input_width * input_height * input_depth;
  int p = window_height;
  int q = window_width;
  int r = input_depth;
  int m = input_height;
  int n = input_width;

  elina_dimperm_t* dp = elina_dimperm_alloc(dim);
  for (int i = 0; i < input_height; i++) {
    for (int j = 0; j < input_width; j++) {
      for (int k = 0; k < input_depth; k++) {
        int row = r * p * q * ((n / q) * (i / p) + (j / q)) + p * q * k +
          q * (i % p) + j % q;
        int col = n * r * i + r * j + k;
        dp->dim[col] = row;
      }
    }
  }
  Powerset z = pow.permute_dimensions(dp);
  elina_dimperm_free(dp);

  // Use a bunch of case expressions
  for (int i = 0; i < (m / p) * (n / q) * r; i++) {
    // Note that each i collapses it's segment of the input vector so as long
    // as we traverse these i's in order we can just take the p * q elements
    // starting at index i
    std::vector<elina_lincons0_array_t> maxes;
    for (int j = 0; j < p * q; j++) {
      elina_lincons0_array_t max = elina_lincons0_array_make(p * q - 1);
      int ind = 0;
      for (int k = 0; k < p * q; k++) {
        if (k == j) {
          continue;
        }
        elina_linexpr0_t* gt = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 2);
        // x_j >= x_k --> x_j - x_k >= 0
        elina_linexpr0_set_coeff_scalar_double(gt, i + j, 1.0);
        elina_linexpr0_set_coeff_scalar_double(gt, i + k, -1.0);
        max.p[ind].constyp = ELINA_CONS_SUPEQ;
        max.p[ind].linexpr0 = gt;
        ind++;
      }
      maxes.push_back(max);
    }
    std::vector<Powerset> ps;
    for (int j = 0; j < p * q; j++) {
      // Meet z with each element of maxes
      Powerset t = z.meet_lincons_array(&maxes[j]);
      elina_dimchange_t* dc = elina_dimchange_alloc(0, p * q - 1);
      int ind = 0;
      for (int k = 0; k < p * q; k++) {
        if (k == j) {
          continue;
        }
        dc->dim[ind] = i + k;
        ind++;
      }
      t = t.remove_dimensions(dc);
      elina_dimchange_free(dc);
      ps.push_back(t);
      elina_lincons0_array_clear(&maxes[j]);
    }
    // Join
    z = ps[0];
    for (unsigned int j = 1; j < ps.size(); j++) {
      z = z.join(ps[j]);
    }
  }
  return z;
}

FCLayer::FCLayer() : Layer(0, 0, 0, 0, 0, 0) {}

FCLayer::FCLayer(const Eigen::MatrixXd& w, const Eigen::VectorXd& b) :
    Layer(1, w.cols(), 1, 1, b.size(), 1), weight{w}, bias{b} {
  if (w.rows() != b.size()) {
    throw std::runtime_error("Bad initialization of FCLayer");
  }
}

FCLayer::FCLayer(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, int iw,
    int ih, int id, int ow, int oh, int od) :
  Layer(iw, ih, id, ow, oh, od), weight{w}, bias{b} {
    if (w.rows() != b.size()) {
      throw std::runtime_error("Bad initialization of FCLayer");
    }
  }

LayerType FCLayer::get_type() const {
  return FC;
}

std::vector<Eigen::MatrixXd> FCLayer::evaluate(
    const std::vector<Eigen::MatrixXd>& x) const {
  // Serialize the input tensor
  Eigen::VectorXd x_v(input_width * input_depth * input_height);
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
         x_v(input_width * input_depth * j + input_depth * k + i) = x[i](j,k);
      }
    }
  }

  // Apply the affine transformation
  Eigen::VectorXd out = weight * x_v + bias;

  // Deserialize the output
  std::vector<Eigen::MatrixXd> ret;
  for (int i = 0; i < output_depth; i++) {
    ret.push_back(Eigen::MatrixXd(output_height, output_width));
  }
  for (int i = 0; i < output_depth; i++) {
    for (int j = 0; j < output_height; j++) {
      for (int k = 0; k < output_width; k++) {
        ret[i](j,k) = out(output_width * output_depth * j + output_depth * k + i);
      }
    }
  }
  return ret;
}

std::vector<Eigen::MatrixXd> FCLayer::backpropagate(
    const std::vector<Eigen::MatrixXd>& eval,
    const std::vector<Eigen::MatrixXd>& grad) const {
  Eigen::VectorXd grad_v(output_width * output_depth * output_height);
  for (int i = 0; i < output_depth; i++) {
    for (int j = 0; j < output_height; j++) {
      for (int k = 0; k < output_width; k++) {
        grad_v(output_width * output_depth * j + output_depth * k + i) = grad[i](j,k);
      }
    }
  }
  Eigen::VectorXd out_v = weight.transpose() * grad_v;
  std::vector<Eigen::MatrixXd> out;
  for (int i = 0; i < input_depth; i++) {
    out.push_back(Eigen::MatrixXd(input_height, input_width));
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        out[i](j,k) = out_v(input_width * input_depth * j + input_depth * k + i);
      }
    }
  }
  return out;
}

Powerset FCLayer::propagate_powerset(const Powerset& p) const {
  return p.affine(weight, bias);
}

Filter::Filter(const std::vector<Eigen::MatrixXd>& fs) {
  for (unsigned int i = 0; i < fs.size(); i++) {
    for (unsigned int j = 0; j < fs.size(); j++) {
      if (fs[i].rows() != fs[j].rows() || fs[i].cols() != fs[j].cols()) {
        throw std::runtime_error("Bad construction of Filter");
      }
    }
  }
  data = fs;
}

int Filter::get_depth() const {
  return data.size();
}

int Filter::get_width() const {
  if (data.size() > 0) {
    return data[0].cols();
  } else {
    return 0;
  }
}

int Filter::get_height() const {
  if (data.size() > 0) {
    return data[0].rows();
  } else {
    return 0;
  }
}

double Filter::dot_product(const Filter& other) const {
  if (get_depth() != other.get_depth()) {
    throw std::runtime_error("Dimension mismatch in Filter.dot_product");
  }
  if (get_width() != other.get_width()) {
    throw std::runtime_error("Dimension mismatch in Filter.dot_product");
  }
  if (get_height() != other.get_height()) {
    throw std::runtime_error("Dimension mismatch in Filter.dot_product");
  }

  double sum = 0.0;
  for (int i = 0; i < get_depth(); i++) {
    for (int j = 0; j < get_width(); j++) {
      for (int k = 0; k < get_height(); k++) {
        sum += data[i](k, j) * other.data[i](k, j);
      }
    }
  }
  return sum;
}

ConvLayer::ConvLayer(const std::vector<Filter>& fs,
    const std::vector<double>& bs, int iw, int ih) :
    Layer(iw, ih, fs[0].get_depth(), iw - fs[0].get_width() + 1,
        ih - fs[0].get_height() + 1, fs.size()),
    filter_width{fs[0].get_width()}, filter_height{fs[0].get_height()},
    filter_depth{fs[0].get_depth()}, num_filters(fs.size()), filters(fs),
    biases{bs} {
  int fc_cols = input_width * input_height * input_depth;
  int fc_rows = output_width * output_height * output_depth;

  Eigen::MatrixXd fc_weight = Eigen::MatrixXd::Zero(fc_rows, fc_cols);
  Eigen::VectorXd fc_bias(fc_rows);
  // These computations are taken from the appendix of the AI2 paper
  for (int i = 0; i < input_height - filter_height + 1; i++) {
    for (int j = 0; j < input_width - filter_width + 1; j++) {
      for (int k = 0; k < num_filters; k++) {
        int row = (input_width - filter_width + 1) * num_filters * i +
          num_filters * j + k;
        for (int i2 = 0; i2 < filter_height; i2++) {
          for (int j2 = 0; j2 < filter_width; j2++) {
            for (int k2 = 0; k2 < input_depth; k2++) {
              int col = input_width * input_depth * (i + i2) +
                input_depth * (j + j2) + k2;
              fc_weight(row, col) = filters[k].data[k2](i2, j2);
            }
          }
        }
        fc_bias(row) = biases[k];
      }
    }
  }

  fc = FCLayer(fc_weight, fc_bias, input_width, input_height, input_depth,
      output_width, output_height, output_depth);
}

LayerType ConvLayer::get_type() const {
  return CONV;
}

std::vector<Eigen::MatrixXd> ConvLayer::evaluate(
    const std::vector<Eigen::MatrixXd>& x) const {
  std::vector<Eigen::MatrixXd> ret;
  for (int k = 0; k < num_filters; k++) {
    ret.push_back(Eigen::MatrixXd(output_height, output_width));
    for (int i = 0; i < output_width; i++) {
      for (int j = 0; j < output_height; j++) {
        std::vector<Eigen::MatrixXd> inp;
        for (int k2 = 0; k2 < filter_depth; k2++) {
          inp.push_back(Eigen::MatrixXd(filter_height, filter_width));
          for (int i2 = 0; i2 < filter_width; i2++) {
            for (int j2 = 0; j2 < filter_height; j2++) {
              inp[k2](j2, i2) = x[k2](j + j2, i + i2);
            }
          }
        }
        ret[k](j, i) = filters[k].dot_product(inp) + biases[k];
      }
    }
  }
  return ret;
}

std::vector<Eigen::MatrixXd> ConvLayer::backpropagate(
    const std::vector<Eigen::MatrixXd>& eval,
    const std::vector<Eigen::MatrixXd>& grad) const {
  return fc.backpropagate(eval, grad);
}

Powerset ConvLayer::propagate_powerset(const Powerset& p) const {
  int out_size = output_height * output_width * output_depth;
  int filter_size = filter_height * filter_width * filter_depth;
  elina_dim_t* dims = (elina_dim_t*) malloc(out_size * sizeof(elina_dim_t));
  elina_linexpr0_t** update = (elina_linexpr0_t**) malloc(
      out_size * sizeof(elina_linexpr0_t*));
  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      for (int k = 0; k < output_depth; k++) {
        int row = output_width * output_depth * i + output_depth * j + k;
        dims[row] = row;
        update[row] = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, filter_size);
        for (int i2 = 0; i2 < filter_height; i2++) {
          for (int j2 = 0; j2 < filter_width; j2++) {
            for (int k2 = 0; k2 < filter_depth; k2++) {
              int col = input_width * input_depth * (i + i2) +
                input_depth * (j + j2) + k2;
              elina_linexpr0_set_coeff_scalar_double(update[row], col,
                  filters[k].data[k2](i2, j2));
            }
          }
        }
        elina_linexpr0_set_cst_scalar_double(update[row], biases[k]);
      }
    }
  }
  Powerset ret = p.assign_linexpr_array(dims, update, out_size, out_size);

  free(dims);
  for (int j = 0; j < out_size; j++) {
    elina_linexpr0_free(update[j]);
  }
  free(update);

  return ret;
}

Network::Network() : num_layers{0}, input_width{0}, input_height{0}, input_depth{0},
  output_size{0}, layer_widths{{}}, layer_heights{{}},
  layer_depths{{}}, layers{{}} {}

Network::Network(int nl, int id, int iw, int ih, int os, std::vector<int> lws,
    std::vector<int> lhs, std::vector<int> lds, std::vector<std::shared_ptr<Layer>> ls)
: num_layers{nl}, input_width{iw}, input_height{ih}, input_depth{id}, output_size{os},
  layer_widths(lws), layer_heights(lhs), layer_depths(lds),
  layers(ls) {}

Eigen::VectorXd Network::evaluate(const Eigen::VectorXd& input) const {
  std::vector<Eigen::MatrixXd> in;
  for (int i = 0; i < input_depth; i++) {
    in.push_back(Eigen::MatrixXd(input_height, input_width));
  }
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        in[i](j,k) = input(input_width * input_depth * j + input_depth * k + i);
      }
    }
  }
  for (unsigned int i = 0; i < layers.size(); i++) {
    in = layers[i]->evaluate(in);
    if (i < layers.size() - 1) {
      in = relu(in);
    }
  }
  return in[0].col(0);
}

Eigen::VectorXd Network::gradient(const Eigen::VectorXd& input) const {
  std::vector<Eigen::MatrixXd> in;
  for (int i = 0; i < input_depth; i++) {
    in.push_back(Eigen::MatrixXd(input_height, input_width));
  }
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        in[i](j,k) = input(input_width * input_depth * j + input_depth * k + i);
      }
    }
  }
  std::vector<std::vector<Eigen::MatrixXd>> evaluations;
  for (unsigned int i = 0; i < layers.size(); i++) {
    in = layers[i]->evaluate(in);
    evaluations.push_back(in);
    if (i < layers.size() - 1) {
      in = relu(in);
    }
  }

  std::vector<std::vector<Eigen::MatrixXd>>::iterator it = evaluations.end() -1;
  for (int i = num_layers - 1; i >= 0; i--) {
    if (i < num_layers - 1) {
      for (int j = 0; j < layer_depths[i+1]; j++) {
        for (int k = 0; k < layer_heights[i+1]; k++) {
          for (int l = 0; l < layer_widths[i+1]; l++) {
            if ((*it)[j](k,l) < 0.0) {
              in[j](k,l) = 0.0;
            }
          }
        }
      }
    }
    it--;
    in = layers[i]->backpropagate(*it, in);
  }

  Eigen::VectorXd in_v(input_width * input_depth * input_height);
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        in_v(input_width * input_depth * j + input_depth * k + i) = in[i](j,k);
      }
    }
  }
  return in_v;
}

Eigen::VectorXd parse_vector(std::string s) {
  // Remove the [] around the vector
  s.erase(0, 1);
  s.erase(s.end() - 1, s.end());

  std::stringstream ss(s);
  std::string tok;
  std::vector<double> elems;
  // Split s on commas
  while (getline(ss, tok, ',')) {
    if (tok[0] == ' ') {
      tok.erase(0, 1);
    }
    elems.push_back(atof(tok.c_str()));
  }

  Eigen::VectorXd b(elems.size());
  for (unsigned int i = 0; i < elems.size(); i++) {
    b(i) = elems[i];
  }

  return b;
}

Eigen::MatrixXd parse_matrix(std::string s) {
  // Remove [[ and ]] from the input string
  s.erase(0, 2);
  s.erase(s.end() - 2, s.end());

  std::stringstream ss(s);
  std::string tok;
  std::vector<Eigen::VectorXd> rows;
  while (getline(ss, tok, ']')) {
    // Erase ", [" from the beginning of tok
    if (tok[0] == ',') {
      tok.erase(0, 3);
    }
    Eigen::VectorXd b = parse_vector("[" + tok + "]");
    rows.push_back(b);
  }

  Eigen::MatrixXd m(rows.size(), rows[0].size());
  for (unsigned int i = 0; i < rows.size(); i++) {
    for (int j = 0; j < rows[i].size(); j++) {
      m(i,j) = rows[i](j);
    }
  }

  return m;
}

std::vector<Filter> parse_filters(std::string s,
    int num_filters, int kernel_width, int kernel_height, int kernel_depth) {
  // x[a][b][c][d] is the element at row a, column b, depth c in filter d
  double data[kernel_height][kernel_width][kernel_depth][num_filters];
  int i = 0, j = 0, k = 0, l = -1;
  // Remove all the opening brackets
  s.erase(0, 4);
  std::stringstream ss(s);
  std::string tok;
  while (getline(ss, tok, ',')) {
    while (tok[0] == ' ') {
      tok.erase(0, 1);
    }
    l++;
    if (tok[0] == '[') {
      k++;
      l = 0;
      tok.erase(0, 1);
    }
    if (tok[0] == '[') {
      j++;
      k = 0;
      tok.erase(0, 1);
    }
    if (tok[0] == '[') {
      i++;
      j = 0;
      tok.erase(0, 1);
    }
    while (tok.back() == ']' || tok.back() == ' ') {
      tok.erase(tok.end() - 1, tok.end());
    }
    data[i][j][k][l] = atof(tok.c_str());
  }
  std::vector<Filter> ret;
  for (l = 0; l < num_filters; l++) {
    std::vector<Eigen::MatrixXd> filter;
    for (k = 0; k < kernel_depth; k++) {
      Eigen::MatrixXd weight(kernel_height, kernel_width);
      for (i = 0; i < kernel_height; i++) {
        for (j = 0; j < kernel_width; j++) {
          weight(i,j) = data[i][j][k][l];
        }
      }
      filter.push_back(weight);
    }
    ret.push_back(Filter(filter));
  }
  return ret;
}

std::vector<Powerset> Network::propagate_powerset(const Powerset& p) const {
  Powerset z = p;
  std::vector<Powerset> ret;
  ret.push_back(z);
  struct timespec start, end;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
  for (unsigned int i = 0; i < layers.size(); i++) {
    z = layers[i]->propagate_powerset(z);
    ret.push_back(z);
    //std::cout << "Before ReLU in Layer: " << i << std::endl;
    //for (unsigned int i = 0; i < z.disjuncts.size(); i++) {
    //  elina_abstract0_fprint(stdout, z.disjuncts[i]->man, z.disjuncts[i]->value, NULL);
    //}
    //std::cout << std::endl;
    if (i < layers.size() - 1 && layers[i]->get_type() != MP) {
      z = z.relu();
    }
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    double elapsed = (int) (end.tv_sec - start.tv_sec);
    if (elapsed > 1000) {
      throw std::runtime_error("");
    }
    //std::cout << "After Layer: " << i << std::endl;
    //for (unsigned int i = 0; i < z.disjuncts.size(); i++) {
    //  elina_abstract0_fprint(stdout, z.man, z.disjuncts[i], NULL);
    //}
    //std::cout << std::endl;
  }
  return ret;
}

std::vector<Eigen::MatrixXd> Network::get_weights() const {
  std::vector<Eigen::MatrixXd> ret;
  for (std::shared_ptr<Layer> l : layers) {
    //if (l->get_type() != FC) {
    //  throw std::runtime_error("Non-fully connected layer encountered in get_weights");
    //}
    if (l->get_type() == FC) {
      std::shared_ptr<FCLayer> fcl = std::static_pointer_cast<FCLayer>(l);
      ret.push_back(fcl->weight);
    } else if (l->get_type() == CONV) {
      std::shared_ptr<ConvLayer> cl = std::static_pointer_cast<ConvLayer>(l);
      ret.push_back(cl->fc.weight);
    } else {
      ret.push_back(Eigen::MatrixXd(0, 0));
    }
  }
  return ret;
}

std::vector<Eigen::VectorXd> Network::get_biases() const {
  std::vector<Eigen::VectorXd> ret;
  for (std::shared_ptr<Layer> l : layers) {
    if (l->get_type() != FC) {
      throw std::runtime_error("Non-fully connected layer encountered in get_biases");
    }
    std::shared_ptr<FCLayer> fcl = std::static_pointer_cast<FCLayer>(l);
    ret.push_back(fcl->bias);
  }
  return ret;
}

std::vector<int> Network::get_layer_sizes() const {
  std::vector<int> ret;
  for (std::shared_ptr<Layer> l : layers) {
    ret.push_back(l->get_input_width() * l->get_input_height() * l->get_input_depth());
  }
  std::shared_ptr<Layer> l = layers.back();
  if (l->get_type() != FC) {
    throw std::runtime_error("Non-fully connected layer encountered in get_layer_sizes");
  }
  std::shared_ptr<FCLayer> fcl = std::static_pointer_cast<FCLayer>(l);
  ret.push_back(fcl->get_output_width() * fcl->get_output_height() * fcl->get_output_depth());
  return ret;
}

// Read one of the AI^2 benchmark networks
Network read_network(std::string filename) {
  int num_layers = 0;
  int input_width = 0;
  int input_height = 0;
  int output_size = 0;
  std::vector<int> layer_widths;
  std::vector<int> layer_heights;
  std::vector<int> layer_depths;
  std::vector<std::shared_ptr<Layer>> layers;
  std::ifstream file(filename.c_str());
  std::string line;
  while (getline(file, line)) {
    if (line == "ReLU") {
      getline(file, line);
      // Now line holds the weight for this layer
      Eigen::MatrixXd m = parse_matrix(line);
      if (num_layers == 0) {
        input_width = 1;
        input_height = m.cols();
      }
      getline(file, line);
      Eigen::VectorXd b = parse_vector(line);
      num_layers++;
      if (num_layers == 1) {
        layer_widths.push_back(1);
        layer_heights.push_back(m.cols());
        layer_depths.push_back(1);
      }
      layer_widths.push_back(1);
      layer_heights.push_back(m.rows());
      layer_depths.push_back(1);
      if (num_layers == 1) {
        layers.push_back(std::shared_ptr<Layer>(new FCLayer(m, b)));
      } else {
        layers.push_back(std::shared_ptr<Layer>(new FCLayer(m, b,
                layer_widths[num_layers-1], layer_heights[num_layers-1],
                layer_depths[num_layers-1], 1, m.rows(), 1)));
      }
    } else if (line == "Conv2D") {
      getline(file, line);
      int num_filters = 0;
      int kernel_width = 0, kernel_height = 0;
      int in_width = 0, in_height = 0, in_depth = 0;
      size_t old_pos = 0;
      size_t pos = 0;
      while (pos < line.size()) {
        int bcount = 0;
        while (pos < line.size() && (line[pos] != ',' || bcount != 0)) {
          if (line[pos] == '[') {
            bcount++;
          } else if (line[pos] == ']') {
            bcount--;
          }
          pos++;
        }
        std::string tok = line.substr(old_pos, pos - old_pos);
        pos++;
        old_pos = pos;
        while (tok[0] == ' ' || tok[0] == ',') {
          tok.erase(0, 1);
        }
        if (tok == "ReLU") {
          continue;
        }
        size_t ind = tok.find("=");
        if (ind == std::string::npos) {
          throw std::runtime_error("Failed to parse network");
        }
        std::string name = tok.substr(0, ind);
        std::string val = tok.substr(ind+1, std::string::npos);
        if (name == "filters") {
          num_filters = stoi(val);
        } else if (name == "kernel_size") {
          Eigen::VectorXd v = parse_vector(val);
          kernel_height = v(0);
          kernel_width = v(1);
        } else if (name == "input_shape") {
          Eigen::VectorXd v = parse_vector(val);
          in_height = v(0);
          in_width = v(1);
          in_depth = v(2);
        } else {
          throw std::runtime_error("Unrecognized option when reading network");
        }
      }
      getline(file, line);
      std::string line2;
      std::vector<Filter> fs = parse_filters(line, num_filters,
          kernel_width, kernel_height, in_depth);
      getline(file, line);
      Eigen::VectorXd b = parse_vector(line);
      std::vector<double> bs;
      for (int i = 0; i < b.size(); i++) {
        bs.push_back(b(i));
      }
      if (num_layers == 0) {
        input_width = in_width;
        input_height = in_height;
      }
      num_layers++;
      if (num_layers == 1) {
        layer_widths.push_back(in_width);
        layer_heights.push_back(in_height);
        layer_depths.push_back(in_depth);
      }
      layer_widths.push_back(in_width - kernel_width + 1);
      layer_heights.push_back(in_height - kernel_height + 1);
      layer_depths.push_back(num_filters);
      layers.push_back(std::shared_ptr<Layer>(new ConvLayer(fs, bs, in_width, in_height)));
    } else if (line == "MaxPooling2D") {
      getline(file, line);
      int pool_width = 0, pool_height = 0;
      int in_width = 0, in_height = 0, in_depth = 0;
      size_t old_pos = 0;
      size_t pos = 0;
      while (pos < line.size()) {
        int bcount = 0;
        while (pos < line.size() && (line[pos] != ',' || bcount != 0)) {
          if (line[pos] == '[') {
            bcount++;
          } else if (line[pos] == ']') {
            bcount--;
          }
          pos++;
        }
        std::string tok = line.substr(old_pos, pos - old_pos);
        pos++;
        old_pos = pos;
        while (tok[0] == ' ' || tok[0] == ',') {
          tok.erase(0, 1);
        }
        if (tok == "ReLU") {
          continue;
        }
        size_t ind = tok.find("=");
        if (ind == std::string::npos) {
          throw std::runtime_error("Failed to parse network");
        }
        std::string name = tok.substr(0, ind);
        std::string val = tok.substr(ind+1, std::string::npos);
        if (name == "pool_size") {
          Eigen::VectorXd v = parse_vector(val);
          pool_height = v(0);
          pool_width = v(1);
        } else if (name == "input_shape") {
          Eigen::VectorXd v = parse_vector(val);
          in_height = v(0);
          in_width = v(1);
          in_depth = v(2);
        } else {
          throw std::runtime_error("Unrecognized option when reading network");
        }
      }
      if (num_layers == 0) {
        input_width = in_width;
        input_height = in_height;
      }
      num_layers++;
      if (num_layers == 1) {
        layer_widths.push_back(in_width);
        layer_heights.push_back(in_height);
        layer_depths.push_back(in_depth);
      }
      layer_widths.push_back(in_width / pool_width);
      layer_heights.push_back(in_height / pool_height);
      layer_depths.push_back(layer_depths.back());
      layers.push_back(std::shared_ptr<Layer>(
            new MaxPoolLayer(pool_width, pool_height, in_width, in_height, in_depth)));
    } else {
      std::cout << line << std::endl;
      throw std::runtime_error("Unknown layer type");
    }
  }
  output_size = layers.back()->get_output_height();
  return Network(num_layers, 1, input_width, input_height, output_size,
      layer_widths, layer_heights, layer_depths, layers);
}

void print_bounding_box(const Powerset& p) {
  elina_interval_t** box = p.bounding_box();
  for (int i = 0; i < p.dims(); i++) {
    double l, u;
    elina_double_set_scalar(&l, box[i]->inf, MPFR_RNDN);
    elina_double_set_scalar(&u, box[i]->sup, MPFR_RNDN);
    std::cout << i << "\t" << ": " << "[" << l << ", " << u << "]" << std::endl;
    elina_interval_free(box[i]);
  }
  free(box);
}

/*
Eigen::VectorXd Network::backpropagate(
    Eigen::VectorXd input, int max_ind) {
  std::vector<Eigen::MatrixXd> in;
  for (int i = 0; i < input_depth; i++) {
    in.push_back(Eigen::MatrixXd(input_height, input_width));
  }
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        in[i](j,k) = input(input_width * input_depth * j + input_depth * k + i);
      }
    }
  }
  std::vector<std::vector<Eigen::MatrixXd>> evaluations;
  for (unsigned int i = 0; i < layers.size(); i++) {
    in = layers[i]->evaluate(in);
    evaluations.push_back(in);
    if (i < layers.size() - 1) {
      in = relu(in);
    }
  }

  std::vector<Eigen::MatrixXd> out = evaluations.back();
  evaluations.pop_back();
  Eigen::VectorXd scores = out[0].col(0);
  Eigen::VectorXd softmax(scores.size());
  double softmax_total = 0.0;
  for (int i = 0; i < scores.size(); i++) {
    softmax(i) = std::exp(scores(i));
    softmax_total += softmax(i);
  }
  softmax /= softmax_total;
  softmax(max_ind) -= 1;
  std::vector<Eigen::MatrixXd> new_back;
  Eigen::MatrixXd mat(softmax.size(), 1);
  mat.col(0) = softmax;
  new_back.push_back(mat);
  evaluations.push_back(new_back);

  std::vector<std::vector<Eigen::MatrixXd>>::iterator it = evaluations.end() -1;
  for (int i = num_layers - 1; i >= 0; i--) {
    if (i < num_layers - 1) {
      for (int j = 0; j < layer_depths[i+1]; j++) {
        for (int k = 0; k < layer_heights[i+1]; k++) {
          for (int l = 0; l < layer_widths[i+1]; l++) {
            if ((*it)[j](k,l) < 0.0) {
              in[j](k,l) = 0.0;
            }
          }
        }
      }
    }
    it--;
    in = layers[i]->backpropagate(*it, in);
  }

  Eigen::VectorXd in_v(input_width * input_depth * input_height);
  for (int i = 0; i < input_depth; i++) {
    for (int j = 0; j < input_height; j++) {
      for (int k = 0; k < input_width; k++) {
        in_v(input_width * input_depth * j + input_depth * k + i) = in[i](j,k);
      }
    }
  }

  return in_v;
}
*/
