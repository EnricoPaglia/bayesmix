#include "nnig_python_hierarchy.h"

#include <google/protobuf/stubs/casts.h>

#include <Eigen/Dense>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "src/utils/rng.h"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;
using namespace py::literals;

extern py::module_ numpy_random;
extern py::object py_engine;
extern py::object py_gen;

//! PYTHON
double NNIG_PYTHONHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    return stan::math::normal_lpdf(datum(0), state.generic_state[0], sqrt(state.generic_state[1]));
}

//! PYTHON
double NNIG_PYTHONHierarchy::marg_lpdf(const NNIG_PYTHON::Hyperparams &params,
                                       const Eigen::RowVectorXd &datum) const {
    double sig_n = sqrt(params.generic_hypers[3] * (params.generic_hypers[1] + 1) /
                        (params.generic_hypers[2] * params.generic_hypers[1]));
    return stan::math::student_t_lpdf(datum(0), 2 * params.generic_hypers[2], params.generic_hypers[0],
                                      sig_n);
}

//! PYTHON
void NNIG_PYTHONHierarchy::initialize_state() {
    state.generic_state.clear();
    state.generic_state.push_back(hypers->generic_hypers[0]);
    state.generic_state.push_back(hypers->generic_hypers[3] / (hypers->generic_hypers[2] + 1));
}

//! C++
void NNIG_PYTHONHierarchy::initialize_hypers() {
    if (prior->has_values()) {
        // Set values
        hypers->generic_hypers.clear();
        int size = prior->values().size();
        for(int i = 0; i < size; ++i){
            hypers->generic_hypers.push_back((prior->values().data())[i]);
        }
    }
}

//! PYTHON
void NNIG_PYTHONHierarchy::update_hypers(
        const std::vector <bayesmix::AlgorithmState::ClusterState> &states) {
    auto &rng = bayesmix::Rng::Instance().get();
    if (prior->has_values())
        return;
}

//! PYTHON
NNIG_PYTHON::State NNIG_PYTHONHierarchy::draw(const NNIG_PYTHON::Hyperparams &params) {
    auto &rng = bayesmix::Rng::Instance().get();
    NNIG_PYTHON::State out;
    out.generic_state.push_back(stan::math::normal_rng(params.generic_hypers[0] ,
                                                       sqrt(state.generic_state[1] / params.generic_hypers[1]), rng));
    out.generic_state.push_back(stan::math::inv_gamma_rng(params.generic_hypers[2], params.generic_hypers[3], rng));

    return out;
}

//! ?
void NNIG_PYTHONHierarchy::update_summary_statistics(const Eigen::RowVectorXd &datum,
                                                     const bool add) {
    if (add) {
        data_sum += datum(0);
        data_sum_squares += datum(0) * datum(0);
    } else {
        data_sum -= datum(0);
        data_sum_squares -= datum(0) * datum(0);
    }
}

//! ?
void NNIG_PYTHONHierarchy::clear_summary_statistics() {
    data_sum = 0;
    data_sum_squares = 0;
}

//! PYTHON
NNIG_PYTHON::Hyperparams NNIG_PYTHONHierarchy::compute_posterior_hypers() const {
    // Initialize relevant variables
    if (card == 0) {  // no update possible
        return *hypers;
    }
    // Compute posterior hyperparameters
    NNIG_PYTHON::Hyperparams post_params;
    double y_bar = data_sum / (1.0 * card);  // sample mean
    double ss = data_sum_squares - card * y_bar * y_bar;
    post_params.generic_hypers.push_back((hypers->generic_hypers[1] * hypers->generic_hypers[0] + data_sum) /
                       (hypers->generic_hypers[1] + card));
    post_params.generic_hypers.push_back(hypers->generic_hypers[1] + card);
    post_params.generic_hypers.push_back(hypers->generic_hypers[2] + 0.5 * card);
    post_params.generic_hypers.push_back(hypers->generic_hypers[3] + 0.5 * ss +
                        0.5 * hypers->generic_hypers[1] * card *
                        (y_bar - hypers->generic_hypers[0]) * (y_bar - hypers->generic_hypers[0]) /
                        (card + hypers->generic_hypers[1]));
    return post_params;
}

//! C++
void NNIG_PYTHONHierarchy::set_state_from_proto(
        const google::protobuf::Message &state_) {
    auto &statecast = downcast_state(state_);
    int size = statecast.vector_state().size();
    std::vector<double> aux_v{};
    for (int i = 0; i < size; ++i) {
        aux_v.push_back((statecast.vector_state().data())[i]);
    }
    state.generic_state = aux_v;
    set_card(statecast.cardinality());
}

//! C++
std::shared_ptr <bayesmix::AlgorithmState::ClusterState>
NNIG_PYTHONHierarchy::get_state_proto() const {
    bayesmix::VectorState state_;
    state_.set_size(state.generic_state.size());
    *state_.mutable_data() = {state.generic_state.data(), state.generic_state.data() + state.generic_state.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
    out->mutable_vector_state()->CopyFrom(state_);
    return out;
}

//! C++
void NNIG_PYTHONHierarchy::set_hypers_from_proto(
        const google::protobuf::Message &hypers_) {
    auto &hyperscast = downcast_hypers(hypers_).nnig_python_state();
    int size = hyperscast.data().size();
    std::vector<double> aux_v{};
    for (int i = 0; i < size; ++i) {
        aux_v.push_back((hyperscast.data())[i]);
    }
    hypers->generic_hypers = aux_v;
}

//! C++
std::shared_ptr <bayesmix::AlgorithmState::HierarchyHypers>
NNIG_PYTHONHierarchy::get_hypers_proto() const {
    bayesmix::GenericDistribution hypers_;
    hypers_.set_size(hypers->generic_hypers.size());
    *hypers_.mutable_data() = {hypers->generic_hypers.data(), hypers->generic_hypers.data() + hypers->generic_hypers.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
    out->mutable_nnig_python_state()->CopyFrom(hypers_);
    return out;
}