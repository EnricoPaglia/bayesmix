#ifndef BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_
#define BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <cassert>
#include <memory>

#include "../../proto/cpp/hierarchy_prior.pb.h"
#include "../../proto/cpp/marginal_state.pb.h"
#include "base_hierarchy.hpp"

//! Normal Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchy, i.e. a cluster, whose univariate data
//! are distributed according to a normal likelihood, the parameters of which
//! have a Normal-InverseGamma centering distribution. That is:
//!           phi = (mu,sig)     (state);
//! f(x_i|mu,sig) = N(mu,sig^2)  (data likelihood);
//!    (mu,sig^2) ~ G            (unique values distribution);
//!             G ~ MM           (mixture model);
//!            G0 = N-IG         (centering distribution).
//! state[0] = mu is called location, and state[1] = sig is called scale. The
//! state hyperparameters, contained in the Hypers object, are (mu_0, lambda0,
//! alpha0, beta0), all scalar values. Note that this hierarchy is conjugate,
//! thus the marginal and the posterior distribution are available in closed
//! form and Neal's algorithm 2 may be used with it.

class NNIGHierarchy : public BaseHierarchy {
 public:
  struct State {
    double mean, var;
  };
  struct Hyperparams {
    double mean, var_scaling, shape, scale;
  };

 protected:
  double data_sum = 0;
  double data_sum_squares = 0;
  // STATE
  State state;
  // HYPERPARAMETERS
  std::shared_ptr<Hyperparams> hypers;
  // HYPERPRIOR
  std::shared_ptr<bayesmix::NNIGPrior> prior;

  void clear_data() {
    data_sum = 0;
    data_sum_squares = 0;
    card = 0;
    cluster_data_idx = std::set<int>();
  }

  void update_summary_statistics(const Eigen::VectorXd &datum, bool add) {
    if (add) {
      data_sum += datum(0);
      data_sum_squares += datum(0) * datum(0);
    }
    else {
      data_sum -= datum(0);
      data_sum_squares -= datum(0) * datum(0);
    }
  }

  // AUXILIARY TOOLS
  //! Returns updated values of the prior hyperparameters via their posterior
  Hyperparams normal_invgamma_update();

 public:
  void initialize() override;
  //! Returns true if the hierarchy models multivariate data (here, false)
  bool is_multivariate() const override { return false; }

  void update_hypers(const std::vector<bayesmix::MarginalState::ClusterState>
                         &states) override;

  // DESTRUCTOR AND CONSTRUCTORS
  ~NNIGHierarchy() = default;
  NNIGHierarchy() = default;

  std::shared_ptr<BaseHierarchy> clone() const override {
    auto out = std::make_shared<NNIGHierarchy>(*this);
    out->clear_data();
    return out;
  }

  // EVALUATION FUNCTIONS
  //! Evaluates the log-likelihood of data in a single point
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;
  //! Evaluates the log-likelihood of data in the given points
  Eigen::VectorXd like_lpdf_grid(const Eigen::MatrixXd &data) const override;
  //! Evaluates the log-marginal distribution of data in a single point
  double marg_lpdf(const Eigen::RowVectorXd &datum) const override;
  //! Evaluates the log-marginal distribution of data in the given points
  Eigen::VectorXd marg_lpdf_grid(const Eigen::MatrixXd &data) const override;

  // SAMPLING FUNCTIONS
  //! Generates new values for state from the centering prior distribution
  void draw() override;
  //! Generates new values for state from the centering posterior distribution
  void sample_given_data() override;
  void sample_given_data(const Eigen::MatrixXd &data) override;

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  Hyperparams get_hypers() const { return *hypers; }
  //! \param state_ State value to set
  //! \param check  If true, a state validity check occurs after assignment
  void set_state_from_proto(const google::protobuf::Message &state_) override;
  void set_prior(const google::protobuf::Message &prior_) override;
  void write_state_to_proto(google::protobuf::Message *out) const override;
  void write_hypers_to_proto(google::protobuf::Message *out) const override;

  std::string get_id() const override { return "NNIG"; }
};

#endif  // BAYESMIX_HIERARCHIES_NNIG_HIERARCHY_HPP_
