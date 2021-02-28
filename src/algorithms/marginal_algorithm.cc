#include "marginal_algorithm.h"

#include <Eigen/Dense>
#include <stan/math/prim/fun.hpp>

#include "algorithm_state.pb.h"
#include "base_algorithm.h"
#include "src/collectors/base_collector.h"
#include "src/mixings/marginal_mixing.h"

void MarginalAlgorithm::initialize() {
  BaseAlgorithm::initialize();
  marg_mixing = std::dynamic_pointer_cast<MarginalMixing>(mixing);
}

Eigen::VectorXd MarginalAlgorithm::lpdf_from_state(
    const Eigen::MatrixXd &grid, const Eigen::MatrixXd &hier_covariates,
    const Eigen::MatrixXd &mix_covariates) {
  // Read mixing state
  unsigned int n_data = curr_state.cluster_allocs_size();
  unsigned int n_clust = curr_state.cluster_states_size();
  marg_mixing->set_state_from_proto(curr_state.mixing_state());
  // Initialize estimate containers
  Eigen::MatrixXd lpdf_local(grid.rows(), n_clust + 1);
  Eigen::VectorXd lpdf_final(grid.rows());
  auto temp_hier = unique_values[0]->clone();
  // Loop over grid points
  for (size_t i = 0; i < grid.rows(); i++) {
    // Get mass values for i-th grid point
    double mass_ex = marg_mixing->mass_existing_cluster(
        n_data, true, false, temp_hier, mix_covariates.row(i));
    double mass_new = marg_mixing->mass_new_cluster(
        n_data, true, false, n_clust, mix_covariates.row(i));
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      temp_hier->set_state_from_proto(curr_state.cluster_states(j));
      // Get local, single-point estimate
      lpdf_local(i, j) =
          mass_ex + temp_hier->like_lpdf(grid.row(i), hier_covariates.row(i));
    }
    lpdf_local(i, n_clust) =
        mass_new + lpdf_marginal_component(temp_hier, grid.row(i),
                                           hier_covariates.row(i))(0);
    // Final estimate for i-th grid point
    lpdf_final(i) = stan::math::log_sum_exp(lpdf_local.row(i));
  }
  return lpdf_final;
}
