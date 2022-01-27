#ifndef BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_METROPOLIS_UPDATER_H_

#include "abstract_updater.h"

template <class DerivedUpdater>
class MetropolisUpdater : public AbstractUpdater {
 public:

  template <typename F>
  void draw(AbstractLikelihood &like, AbstractPriorModel &prior,
            bool update_params, F& target_lpdf) {
    Eigen::VectorXd curr_state = like.get_unconstrained_state();
    Eigen::VectorXd prop_state = static_cast<DerivedUpdater*>(this)->sample_proposal(
        curr_state, like, prior, target_lpdf);

    double log_arate = like.cluster_lpdf_from_unconstrained(prop_state) -
                       like.cluster_lpdf_from_unconstrained(curr_state) +
                       static_cast<DerivedUpdater*>(this)->proposal_lpdf(
                           curr_state, prop_state, like, prior, target_lpdf) -
                       static_cast<DerivedUpdater*>(this)->proposal_lpdf(
                           prop_state, curr_state, like, prior, target_lpdf);

    auto &rng = bayesmix::Rng::Instance().get();
    if (std::log(stan::math::uniform_rng(0, 1, rng)) < log_arate) {
      like.set_state_from_unconstrained(prop_state);
    }
  }
   
//   template <typename F>
//   virtual Eigen::VectorXd sample_proposal(Eigen::VectorXd curr_state,
//                                           AbstractLikelihood &like,
//                                           AbstractPriorModel &prior,
//                                           F& target_lpdf) = 0;
   
//   template <typename F> 
//   virtual double proposal_lpdf(Eigen::VectorXd prop_state,
//                                Eigen::VectorXd curr_state,
//                                AbstractLikelihood &like,
//                                AbstractPriorModel &prior,
//                                F& target_lpdf) = 0;
};

#endif
