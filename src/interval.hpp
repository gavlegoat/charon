//
// Created by shankara on 10/12/18.
//

#ifndef _INTERVAL_H_
#define _INTERVAL_H_

#include <cstdlib>
#include <vector>
#include <Eigen/Dense>

/**
 * A hyperinterval class used to define properties. A robustness property
 * consists of an instance of this class along with an integer which is the
 * target output.
 */
class Interval {
  public:
    /** Lower bounds for each dimension. */
    Eigen::VectorXd lower;
    /** Upper bounds for each dimension. */
    Eigen::VectorXd upper;
    /** Holds the indices of each dimension which has positive width. */
    std::vector<int> posDims;

    /** Default constructor. Leaves lower and upper empty. */
    Interval();

    /**
     * Constructor taking the interval bounds.
     *
     * \param l The lower bounds for each dimension.
     * \param u The upper bounds for each dimension.
     */
    Interval(Eigen::VectorXd l, Eigen::VectorXd u);

    /**
     * Read an interval from a file. The file format consists of a number of
     * lines, each of which has the form `[l, u]`. The result is an interval
     * with one dimension per line where the bounds for each interval are given
     * in the corresponding line of the file.
     *
     * \param filename The name of the file to read the interval from.
     */
    Interval(std::string filename);

    /**
     * Convert this interval to a type useful for ELINA.
     *
     * \return An ELINA interval equivalent to this.
     */
    elina_interval_t** get_elina_interval() const;

    /**
     * Set the bounds of this interval.
     *
     * \param l The new lower bounds of the interval.
     * \param u The new upper bounds of the interval.
     */
    void set_bounds(Eigen::VectorXd l, Eigen::VectorXd u);

    /**
     * Find the center of this interval.
     *
     * \return The center of this interval as an Eigen vector.
     */
    Eigen::VectorXd get_center() const;

    /**
     * Find the dimension which is the longest. That is, find the value of i
     * which maximizes `upper(i) - lower(i)`.
     *
     * \return The index of the longest dimension of this interval.
     */
    int longest_dim() const;

    /**
     * Find the dimension along which the counterexample is farthest from the
     * center of the interval.
     *
     * \return The index of the largest dimension.
     */
    int longest_dim(const Eigen::VectorXd &counterexample) const;

    /**
     * Find the average dimension length of this interval.
     *
     * \return The average length of the dimensions.
     */
    double average_len() const;
};

#endif //PROJECT_INTERVAL_H
