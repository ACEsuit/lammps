/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mace/kk,PairMACEKokkos<LMPDeviceType>);
PairStyle(mace/kk/device,PairMACEKokkos<LMPDeviceType>);
PairStyle(mace/kk/host,PairMACEKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_MACE_KOKKOS_H
#define LMP_PAIR_MACE_KOKKOS_H

#include "pair_mace.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMACEKokkos : public PairMACE {

 public:

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairMACEKokkos(class LAMMPS *);
  ~PairMACEKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 protected:

  int host_flag;
  typedef Kokkos::DualView<F_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<F_FLOAT**, DeviceType> t_fparams;
  t_fparams d_cutsq;

  // new
  Kokkos::View<int64_t*,DeviceType> k_lammps_atomic_numbers;
  Kokkos::View<int64_t*,DeviceType> k_mace_atomic_numbers;
  int mace_atomic_numbers_size;

};
}    // namespace LAMMPS_NS

#endif
#endif
