#ifndef MESSOPS_LIBINT_BRIDGE_HPP
#define MESSOPS_LIBINT_BRIDGE_HPP

#include <Eigen/Dense>
#include <libint2.hpp>
#include <string>
#include <vector>

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Matrix integrate_1body(const libint2::BasisSet &basis, libint2::Operator optype,
                       const std::vector<libint2::Atom> &atoms, int deriv_order,
                       double precision) {
  libint2::Engine engine(optype, basis.max_nprim(), basis.max_l(), deriv_order,
                         precision);
  const auto &buf = engine.results();

  if (optype == libint2::Operator::nuclear) {
    engine.set_params(libint2::make_point_charges(atoms));
  }

  Matrix result = Matrix::Zero(basis.nbf(), basis.nbf());
  const auto &shell2bf = basis.shell2bf();

  // loop over unique shell pairs, exploiting symmetry
  for (auto s1 = 0; s1 != basis.size(); ++s1) {
    auto n1 = basis[s1].size();

    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto n2 = basis[s2].size();
      engine.compute(basis[s1], basis[s2]);
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);

      auto bf1 = shell2bf[s1];
      auto bf2 = shell2bf[s2];
      result.block(bf1, bf2, n1, n2) = buf_mat;

      if (s1 != s2) {
        result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
      }
    }
  }

  return result;
}

#endif // MESSOPS_LIBINT_BRIDGE_HPP
