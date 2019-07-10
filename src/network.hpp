/* Greg Anderson
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

/* A tensor is represented as a `std::vector<Eigen::MatrixXd>`.
 */

/**
 * Compute the element-wise ReLU (max(x,0)) of a vector.
 */
Eigen::VectorXd relu(const Eigen::VectorXd& x);
/**
 * Compute the element-wise tanh of a vector.
 */
Eigen::VectorXd tanh(const Eigen::VectorXd& x);

/**
 * Label the type of a layer, either convolutional, fully connected, or max
 * pooling.
 */
enum LayerType { CONV, FC, MP };

/**
 * One layer of a neural network.
 */
class Layer {
  protected:
    /** The width of the input to this layer. */
    int input_width;
    /** The height of the input to this layer. */
    int input_height;
    /** The depth of the input to this layer. */
    int input_depth;
    /** The width of the input to this layer. */
    int output_width;
    /** The height of the input to this layer. */
    int output_height;
    /** The depth of the input to this layer. */
    int output_depth;

  public:
    /**
     * Construct a new layer with the given dimensions.
     *
     * \param iw The input width.
     * \param ih The input height.
     * \param id The input depth.
     * \param ow The output width.
     * \param oh The output height.
     * \param od The output depth.
     */
    Layer(int iw, int ih, int id, int ow, int oh, int od);

    /**
     * Get the input width.
     *
     * \return The input width.
     */
    inline int get_input_width() const {
      return input_width;
    }

    /**
     * Get the input height.
     *
     * \return The input height.
     */
    inline int get_input_height() const {
      return input_height;
    }

    /**
     * Get the input depth.
     *
     * \return The input depth.
     */
    inline int get_input_depth() const {
      return input_depth;
    }

    /**
     * Get the output width.
     *
     * \return The output width.
     */
    inline int get_output_width() const {
      return output_width;
    }

    /**
     * Get the output height.
     *
     * \return The output height.
     */
    inline int get_output_height() const {
      return output_height;
    }

    /**
     * Get the output depth.
     *
     * \return The output depth.
     */
    inline int get_output_depth() const {
      return output_depth;
    }

    /**
     * Get the type of this layer.
     *
     * \return The type of this layer.
     */
    virtual LayerType get_type() const = 0;

    /**
     * Determine how this layer changes it's input.
     *
     * \param x The input to the layer.
     * \return The output of this layer evaluated on `x`.
     */
    virtual std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>& x) const = 0;

    /**
     * Given the values seen during execution of this layer and gradients at
     * the output, compute the gradient at the input.
     *
     * \param eval The value of the output of the layer.
     * \param grad The gradients at the output of the layer.
     * \return The gradients at the input to the layer.
     */
    virtual std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>& eval,
        const std::vector<Eigen::MatrixXd>& grad) const = 0;

    /**
     * Propogate an abstract value through this layer.
     *
     * \param x An abstract value describing the input.
     * \return An abstract value describing the output.
     */
    virtual Powerset propagate_powerset(const Powerset& x) const = 0;
};

/**
 * A max pooling layer.
 */
class MaxPoolLayer: public Layer {
  public:
    /** The width of the pooling window. */
    int window_width;
    /** The height of the pooling window. */
    int window_height;

    /**
     * Construct a max pooling layer from the window and input dimensions. The
     * output dimensions are uniquely determined by these values.
     *
     * \param ww The width of the pooling window.
     * \param wh The height of the pooling window.
     * \param iw The width of the input tensor.
     * \param ih The height of the input tensor.
     * \param d The depth of the input tensor.
     */
    MaxPoolLayer(int ww, int wh, int iw, int ih, int d);

    LayerType get_type() const override;
    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;
    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;
    Powerset propagate_powerset(const Powerset&) const override;
};

/**
 * A fully connected layer.
 */
class FCLayer: public Layer {
  public:
    /** The weights between each input-output pair. */
    Eigen::MatrixXd weight;
    /** The biases for each output. */
    Eigen::VectorXd bias;

    /** Construct a layer with empty weight and biases. */
    FCLayer();

    /**
     * Construct a layer with the given weight and biases. This constructor
     * computes the input and output layer sizes from the weight matrix and
     * assumes both the input and the output are column vectors.
     *
     * \param w The weights to use in this layer.
     * \param b The biases to use in this layer.
     */
    FCLayer(const Eigen::MatrixXd& w, const Eigen::VectorXd& b);

    /**
     * Construct a layer with the given weight, biases, and dimensions. This
     * constructor does not assume that the input or output is a column vector.
     * Rather, the input tensor is reshaped into a column vector, transformed,
     * and then the output is reshaped into a tensor according to the given
     * sizes.
     *
     * \param w The weights to use for this layer.
     * \param b The biases to use for this layer.
     * \param iw The width of the input tensor.
     * \param ih The height of the input tensor.
     * \param id The depth of the input tensor.
     * \param ow The width of the output tensor.
     * \param oh The height of the output tensor.
     * \param od The depth of the output tensor.
     */
    FCLayer(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, int iw,
        int ih, int id, int ow, int oh, int od);

    LayerType get_type() const override;
    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;
    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;
    Powerset propagate_powerset(const Powerset&) const override;
};

/**
 * A filter for a convolutional layer.
 */
class Filter {
  public:
    /** The coefficients of this filter */
    std::vector<Eigen::MatrixXd> data;

    /**
     * Construct a filter from the given data
     *
     * \param d The coefficients to use.
     */
    Filter(const std::vector<Eigen::MatrixXd>& x);

    /**
     * Get the depth of this filter.
     *
     * \return The depth of the filter.
     */
    int get_depth() const;

    /**
     * Get the width of this filter.
     *
     * \return The width of the filter.
     */
    int get_width() const;

    /**
     * Get the height of this filter.
     *
     * \return The height of the filter.
     */
    int get_height() const;

    /**
     * Compute the dot product of this filter with `other`.
     *
     * \param other The other filter to compute a dot product with
     * \return The dot product of this filter with `other`.
     */
    double dot_product(const Filter& other) const;
};

/**
 * A convolutional layer.
 */
class ConvLayer: public Layer {
  public:
    /** The width of each filter. */
    int filter_width;
    /** The height of each filter. */
    int filter_height;
    /** The depth of each filter. */
    int filter_depth;
    /** The number of filters to use. */
    int num_filters;
    /** The filters for this layer. */
    std::vector<Filter> filters;
    /** The biases for the outputs. */
    std::vector<double> biases;

    /**
     * A fully connected layer which is equivalent to this convolutional layer.
     */
    FCLayer fc;

    /**
     * Construct a layer from the given filters, biases, and input dimensions.
     * The input depth and output dimensions can be extracted from the given
     * values.
     *
     * \param fs The filters for this layer.
     * \param bs The biases for this layer.
     * \param iw The width of the input tensor.
     * \param ih The height of the input tensor.
     */
    ConvLayer(const std::vector<Filter>& fs,
        const std::vector<double>& bs, int iw, int ih);

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
    /** The width of the input to the network. */
    int input_width;
    /** The height of the input to the network. */
    int input_height;
    /** The depth of the input to the network. */
    int input_depth;
    /**
     * The size of the output vector. The network is assumed to be used for
     * classification, so the output tensor is one-dimensional.
     */
    int output_size;
    /** The widths of each layer, including input and output. */
    std::vector<int> layer_widths;
    /** The heights of each layer, including input and output. */
    std::vector<int> layer_heights;
    /** The depths of each layer, including input and output. */
    std::vector<int> layer_depths;
    /** The layers in this network. */
    std::vector<std::shared_ptr<Layer>> layers;

  public:
    /**
     * Create a new network with 0-dimensional input and no data.
     */
    Network();

    /**
     * Create a new network given all the relevant information.
     *
     * \param nl The number of layers.
     * \param id The depth of the input tensor.
     * \param iw The width of the input tensor.
     * \param ih The height of the input tensor.
     * \param os The size of the output vector.
     * \param lws The widths of each layer.
     * \param lhs The heights of each layer.
     * \param lds The depths of each layer.
     * \param ls The layers.
     */
    Network(int nl, int id, int iw, int ih, int os, std::vector<int> lws,
        std::vector<int> lhs, std::vector<int> lds, std::vector<std::shared_ptr<Layer>> ls);

    /**
     * Get the number of layers.
     *
     * \return The number of layers in the network.
     */
    inline int get_num_layers() const {
      return num_layers;
    }

    /**
     * Get the width of the input tensor.
     *
     * \return The width of the input to the network.
     */
    inline int get_input_width() const {
      return input_width;
    }

    /**
     * Get the height of the input tensor.
     *
     * \return The height of the input to the network.
     */
    inline int get_input_height() const {
      return input_height;
    }

    /**
     * Get the depth of the input tensor.
     *
     * \return The depth of the input to the network.
     */
    inline int get_input_depth() const {
      return input_depth;
    }

    /**
     * Get the width of the given layer.
     *
     * \param n The layer index to get the width of.
     * \return The width of layer `n`.
     */
    inline int get_layer_width(int n) const {
      return layer_widths[n];
    }

    /**
     * Get the height of the given layer.
     *
     * \param n The layer index to get the height of.
     * \return The height of layer `n`.
     */
    inline int get_layer_height(int n) const {
      return layer_heights[n];
    }

    /**
     * Get the depth of the given layer.
     *
     * \param n The layer index to get the depth of.
     * \return The depth of layer `n`.
     */
    inline int get_layer_depth(int n) const {
      return layer_depths[n];
    }

    /**
     * Get the total number of input parameters. This is just
     * `input_width * input_height * input_depth`.
     *
     * \return The total number of input parameters.
     */
    inline int get_input_size() const {
      return input_width * input_height * input_depth;
    }

    /**
     * Get the output size.
     *
     * \return The size of the output vector.
     */
    inline int get_output_size() const {
      return output_size;
    }

    /**
     * Get a reference to some layer of the network.
     *
     * \param n The index of the layer to get.
     * \return A pointer to layer `n`.
     */
    inline std::shared_ptr<Layer> get_layer(int n) const {
      return layers[n];
    }

    /**
     * Evaluate the network at a given input point.
     *
     * \param input The serialized input to the network.
     * \return The output when the network is evaluated at `input`.
     */
    Eigen::VectorXd evaluate(const Eigen::VectorXd& input) const;

    /**
     * Evaluate the network and then backpropagate gradients to the input.
     *
     * \param input The serialized input to the network.
     * \return The gradient of the network at the given input.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& input) const;

    /**
     * Given a powerset representing some input space, find a powerset
     * overapproximating the outputs which can be reached from that input
     * space. This method returns a vector of powersets which includes the
     * abstract values after each layer of the network.
     *
     * \param p An abstract value representing an input to the network.
     * \return A powerset showing the output of each layer of the network.
     */
    std::vector<Powerset> propagate_powerset(const Powerset& p) const;

    /**
     * Get the weights of the network. This method is only defined for networks
     * with only fully connected layers.
     *
     * \return The weight of each layer of the network.
     */
    std::vector<Eigen::MatrixXd> get_weights() const;

    /**
     * Get the biases of the network. This method is only defined for networks
     * with only fully connected layers.
     *
     * \return The biases of each layer of the network.
     */
    std::vector<Eigen::VectorXd> get_biases() const;

    /**
     * Get the size of each layer. The size of a layer is that layer's width
     * times its height times its depth.
     *
     * \return A list with the size of each layer.
     */
    std::vector<int> get_layer_sizes() const;
};

/**
 * Read a network from a file. The file format is as follows:
 * For each layer:
 *   A line with the layer type ("ReLU" for fully connected,
 *          "Conv2D", or "MaxPooling2D")
 *   For fully connected layers,
 *     One line with the weights written as a list
 *          with square brackets, i.e.,[[...], ..., [...]]
 *     One line with biases ([...])
 *   For convolutional layers, one line with:
 *     "ReLU, filters=<k>, kernel_size=[<h>, <w>],
 *            input_shape=[<h>, <w>, <d>]"
 *     One line with a 4D list representing a tensor of kernels
 *     One line with biases
 *   For max pooling layers,
 *     One line with "ReLU, pool_size=[<wh>, <ww>],
 *            input_shape=[<h>, <w>, <d>]"
 *
 * \param filename The file to read the network from.
 * \return The parsed network.
 */
Network read_network(std::string filename);

#endif
