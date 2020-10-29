#include "Algorithm.hpp"

//! \param iter Number of the current iteration
//! \return     Protobuf-object version of the current state
bayesmix::MarginalState Algorithm::get_state_as_proto(unsigned int iter) {
  // Transcribe allocations vector
  bayesmix::MarginalState iter_out;
  *iter_out.mutable_cluster_allocs() = {allocations.begin(),
                                        allocations.end()};

  // Transcribe unique values vector
  for (size_t i = 0; i < unique_values.size(); i++) {
    bayesmix::MarginalState::ClusterVal* clusval = iter_out.add_cluster_vals();
    unique_values[i]->get_state_as_proto(clusval);
  }
  return iter_out;
}