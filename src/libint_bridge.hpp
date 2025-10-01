#ifndef MESSOPS_LIBINT_BRIDGE_HPP
#define MESSOPS_LIBINT_BRIDGE_HPP

#include <Eigen/Dense>
#include <libint2.hpp>
#include <string>
#include <vector>

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Matrix integrate_1body(const libint2::BasisSet &basis, libint2::Operator optype,
                       const std::vector<libint2::Atom> &atoms) {
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  Engine engine(optype, basis.max_nprim(), basis.max_l(), /*deriv_order=*/0);
  const auto &buf = engine.results();

  if (optype == Operator::nuclear) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for (const auto &atom : atoms) {
      q.push_back({static_cast<double>(atom.atomic_number),
                   {{atom.x, atom.y, atom.z}}});
    }
    engine.set_params(q);
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
