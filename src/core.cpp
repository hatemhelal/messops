#include <libint2/atom.h>
#include <libint2/basis.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <iostream>
#include <vector>

#include "libint_bridge.hpp"

namespace nb = nanobind;

using Position = nb::ndarray<double, nb::shape<-1, 3>, nb::device::cpu>;
using Z = nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu>;

struct LibInt2Manager {
  LibInt2Manager() { libint2::initialize(); }
  ~LibInt2Manager() { libint2::finalize(); }
};

void init_libint() { static LibInt2Manager instance; }

struct IntegralContext {
  std::vector<libint2::Atom> atoms;
  libint2::BasisSet basis;
  std::string basis_name;

  IntegralContext(const Z &z, const Position &pos,
                  const std::string &basis_name)
      : atoms(), basis(), basis_name(basis_name) {
    init_libint();

    if (pos.shape(1) != 3) {
      throw std::invalid_argument(
          "Position matrix must have 3 columns (x, y, z).");
    }
    if (pos.shape(0) != z.shape(0)) {
      throw std::invalid_argument(
          "Number of atomic numbers must match number of positions.");
    }

    atoms.reserve(pos.shape(0));

    for (size_t i = 0; i < pos.shape(0); ++i) {
      atoms.emplace_back(libint2::Atom{static_cast<int>(z(i)), pos(i, 0),
                                       pos(i, 1), pos(i, 2)});
    }

    basis = libint2::BasisSet(basis_name, atoms, /*throw_if_no_match=*/true);
  }

  Matrix overlap() const {
    return integrate_1body(basis, libint2::Operator::overlap, atoms);
  }

  Matrix kinetic() const {
    return integrate_1body(basis, libint2::Operator::kinetic, atoms);
  }

  Matrix nuclear() const {
    return integrate_1body(basis, libint2::Operator::nuclear, atoms);
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "IntegralContext(\n"
       << "  basis_name : '" << basis_name << "',\n"
       << "  num_atoms : " << atoms.size() << ",\n"
       << "  num_shells : " << basis.size() << ",\n"
       << "  max_nprim : " << basis.max_nprim() << ",\n"
       << "  max_l : " << basis.max_l() << "\n)";
    return ss.str();
  }
};

NB_MODULE(_core, m) {
  nb::class_<IntegralContext>(m, "IntegralContext")
      .def(nb::init<Z, Position, const std::string &>())
      .def("overlap", &IntegralContext::overlap)
      .def("kinetic", &IntegralContext::kinetic)
      .def("nuclear", &IntegralContext::nuclear)
      .def_ro("basis_name", &IntegralContext::basis_name)
      .def_prop_ro(
          "num_atoms",
          [](const IntegralContext &self) { return self.atoms.size(); })
      .def_prop_ro(
          "max_nprim",
          [](const IntegralContext &self) { return self.basis.max_nprim(); })
      .def_prop_ro(
          "max_l",
          [](const IntegralContext &self) { return self.basis.max_l(); })
      .def_prop_ro(
          "num_shells",
          [](const IntegralContext &self) { return self.basis.size(); })
      .def("__repr__", &IntegralContext::to_string);
}
