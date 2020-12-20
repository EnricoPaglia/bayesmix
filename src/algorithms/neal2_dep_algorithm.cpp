#include "neal2_dep_algorithm.hpp"

#include <Eigen/Dense>
#include <memory>
#include <stan/math/prim/fun.hpp>
#include <vector>

#include "../../proto/cpp/marginal_state.pb.h"
#include "../collectors/base_collector.hpp"
#include "../hierarchies/base_hierarchy.hpp"
#include "../hierarchies/dependent_hierarchy.hpp"
#include "../mixings/base_mixing.hpp"
#include "../utils/distributions.hpp"
#include "../utils/rng.hpp"

void Neal2DepAlgorithm::print_startup_message() const {
  std::string msg = "Running Neal2 dependent algorithm with " +
                    unique_values[0]->get_id() + " hierarchies, " +
                    mixing->get_id() + " mixing...";
  std::cout << msg << std::endl;
}

//! \param temp_hier Temporary hierarchy object
//! \return          Vector of evaluation of component on the provided grid
Eigen::VectorXd Neal2DepAlgorithm::lpdf_marginal_component(
    std::shared_ptr<BaseHierarchy> temp_hier, const Eigen::MatrixXd &grid,
    const Eigen::MatrixXd &covariates) { // TODO anything else?
  // Exploit conjugacy of hierarchy
  return temp_hier->marg_lpdf_grid(grid, covariates);
}

void Neal2DepAlgorithm::sample_allocations() {  // TODO anything else?
  // Initialize relevant values
  unsigned int n_data = data.rows();
  int ndata_from_hier = 0;
  // #ifdef DEBUG
  for (auto &clus : unique_values) ndata_from_hier += clus->get_card();
  assert(n_data == ndata_from_hier);
  // #endif
  auto &rng = bayesmix::Rng::Instance().get();

  // Loop over data points
  for (size_t i = 0; i < n_data; i++) {
    // Initialize current number of clusters
    unsigned int n_clust = unique_values.size();
    // Initialize pseudo-flag
    int singleton = (unique_values[allocations[i]]->get_card() <= 1) ? 1 : 0;
    // Remove datum from cluster
    unique_values[allocations[i]]->remove_datum(i, data.row(i));

    // Compute probabilities of clusters in log-space
    Eigen::VectorXd logprobas(n_clust + 1);
    // Loop over clusters
    for (size_t j = 0; j < n_clust; j++) {
      // Probability of being assigned to an already existing cluster
      logprobas(j) = mixing->mass_existing_cluster(unique_values[j],
                                                   n_data - 1, true, true) +
                     unique_values[j]->like_lpdf(data.row(i));
    }
    // Further update with marginal component
    logprobas(n_clust) =
        mixing->mass_new_cluster(n_clust, n_data - 1, true, true) +
        unique_values[0]->marg_lpdf(data.row(i));

    // Draw a NEW value for datum allocation
    unsigned int c_new =
        bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);
    unsigned int c_old = allocations[i];

    if (c_new == n_clust) {
      auto new_unique = std::dynamic_pointer_cast<DependentHierarchy>(
          unique_values[0]->clone());
      new_unique->add_datum(i, data.row(i), covariates.row(i));
      // Generate new unique values with posterior sampling
      new_unique->sample_given_data();
      unique_values.push_back(new_unique);
      allocations[i] = unique_values.size() - 1;
    } else {
      allocations[i] = c_new;
      auto unique_cast =
          std::dynamic_pointer_cast<DependentHierarchy>(unique_values[c_new]);
      unique_cast->add_datum(i, data.row(i), covariates.row(i));
    }
    if (singleton) {
      // Relabel allocations so that they are consecutive numbers
      for (auto &c : allocations) {
        if (c > c_old) {
          c -= 1;
        }
      }
      unique_values.erase(unique_values.begin() + c_old);
    }
  }
}

void Neal2DepAlgorithm::sample_unique_values() {
  for (auto &clus : unique_values) clus->sample_given_data();
}
