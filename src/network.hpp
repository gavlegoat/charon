/* Greg Anderson
 *
 * Define a neural network.
 */

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "powerset.hpp"

#include <cstdlib>
#include <string>
#include <vector>
#include <memory>

#include <Eigen/Dense>

/**
 * Compute the element-wise ReLU (max(x,0)) of a vector.
 */
Eigen::VectorXd relu(const Eigen::VectorXd& x);
/**
 * Compute the element-wise tanh of a vector.
 */
Eigen::VectorXd tanh(const Eigen::VectorXd& x);

enum LayerType { CONV, FC, MP };

class Layer {
  protected:
    int input_width;
    int input_height;
    int input_depth;
    int output_width;
    int output_height;
    int output_depth;

  public:
    Layer(int, int,  int, int, int, int);
    int get_input_width() const;
    int get_input_height() const;
    int get_input_depth() const;
    int get_output_height() const;
    int get_output_width() const;
    int get_output_depth() const;
    virtual LayerType get_type() const = 0;
    virtual std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const = 0;
    virtual std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const = 0;
    virtual Powerset propagate_powerset(const Powerset&) const = 0;
};

class MaxPoolLayer: public Layer {
  public:
    int window_width;
    int window_height;

    MaxPoolLayer(int, int, int, int, int);
    LayerType get_type() const override;
    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;
    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;
    Powerset propagate_powerset(const Powerset&) const override;
};

class FCLayer: public Layer {
  public:
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    FCLayer();
    FCLayer(const Eigen::MatrixXd&, const Eigen::VectorXd&);
    FCLayer(const Eigen::MatrixXd&, const Eigen::VectorXd&, int iw,
        int ih, int id, int ow, int oh, int od);
    LayerType get_type() const override;
    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;
    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;
    Powerset propagate_powerset(const Powerset&) const override;
};

class Filter {
  public:
    std::vector<Eigen::MatrixXd> data;
    Filter(const std::vector<Eigen::MatrixXd>&);
    int get_depth() const;
    int get_width() const;
    int get_height() const;
    double dot_product(const Filter& other) const;
};

class ConvLayer: public Layer {
  public:
    int filter_width;
    int filter_height;
    int filter_depth;
    int num_filters;
    std::vector<Filter> filters;
    std::vector<double> biases;

    FCLayer fc;
    ConvLayer(const std::vector<Filter>&,
        const std::vector<double>&, int, int);
    LayerType get_type() const override;
    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;
    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;
    Powerset propagate_powerset(const Powerset&) const override;
};

/**
 * A feedforward network.
 */
class Network {
  private:
    /** The number of layers in the network, excluding the input layer. */
    int num_layers;
    int input_width;
    int input_height;
    int input_depth;
    int output_size;
    std::vector<int> layer_widths;
    std::vector<int> layer_heights;
    std::vector<int> layer_depths;
    std::vector<std::shared_ptr<Layer>> layers;

  public:
    Network();
    /**
     * Create a new network given all the relevant information.
     */
    Network(int nl, int id, int iw, int ih, int os, std::vector<int> lws,
        std::vector<int> lhs, std::vector<int> lds, std::vector<std::shared_ptr<Layer>> ls);
    /**
     * Get the number of layers.
     */
    int get_num_layers() const;
    int get_input_width() const;
    int get_input_height() const;
    int get_input_depth() const;
    int get_layer_width(int n) const;
    int get_layer_height(int n) const;
    int get_layer_depth(int n) const;
    int get_input_size() const;
    std::shared_ptr<Layer> get_layer(int) const;
    /**
     * Get the output size.
     */
    int get_output_size() const;
    /**
     * Evaluate the network at a given input point.
     */
    Eigen::VectorXd evaluate(const Eigen::VectorXd& input) const;
    Eigen::VectorXd gradient(const Eigen::VectorXd& input) const;
    /**
     * Given a powerset representing some input space, find a powerset
     * overapproximating the outputs which can be reached from that input
     * space.
     */
    std::vector<Powerset> propagate_powerset(const Powerset& p) const;

    std::vector<Eigen::MatrixXd> get_weights() const;
    std::vector<Eigen::VectorXd> get_biases() const;
    std::vector<int> get_layer_sizes() const;

    Eigen::VectorXd backpropagate(
        Eigen::VectorXd, int max_ind);


};

/**
 * Read a network from a file in AI^2 format.
 */
Network read_network(std::string filename);

#endif
