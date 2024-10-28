/* ----------------------------------------------------------------------
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

#include "pair_mace_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::PairMACEKokkos(LAMMPS *lmp) : PairMACE(lmp)
{
  no_virial_fdotr_compute = 1;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::~PairMACEKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag,0);

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  if (atom->nlocal != list->inum)
    error->all(FLERR, "ERROR: nlocal != inum.");
  if (domain_decomposition && (atom->nghost != list->gnum))
    error->all(FLERR, "ERROR: nghost != gnum.");
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "ERROR: mace/kokkos eflag_atom and/or vflag_atom not implemented.");

  int nlocal = atom->nlocal;
  auto r_max_squared = this->r_max_squared;
  auto h0 = domain->h[0];
  auto h1 = domain->h[1];
  auto h2 = domain->h[2];
  auto h3 = domain->h[3];
  auto h4 = domain->h[4];
  auto h5 = domain->h[5];
  auto hinv0 = domain->h_inv[0];
  auto hinv1 = domain->h_inv[1];
  auto hinv2 = domain->h_inv[2];
  auto hinv3 = domain->h_inv[3];
  auto hinv4 = domain->h_inv[4];
  auto hinv5 = domain->h_inv[5];

  auto _k_lammps_atomic_numbers = k_lammps_atomic_numbers;
  auto _k_mace_atomic_numbers = k_mace_atomic_numbers;
  auto _mace_atomic_numbers_size = mace_atomic_numbers_size;

  // atom map
  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<DeviceType>();

  auto x = atomKK->k_x.view<DeviceType>();
//  c_x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();


  // ----- positions -----
  int n_nodes;
  if (domain_decomposition) {
    n_nodes = atom->nlocal + atom->nghost;
  } else {
    // normally, ghost atoms are included in the graph as independent
    // nodes, as required when the local domain does not have PBC.
    // however, in no_domain_decomposition mode, ghost atoms are simply
    // shifted versions of local atoms.
    n_nodes = atom->nlocal;
  }
  auto k_positions = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_positions", n_nodes);
  Kokkos::parallel_for("PairMACEKokkos: Fill k_positions.", n_nodes, KOKKOS_LAMBDA (const int i) {
    k_positions(i,0) = x(i,0);
    k_positions(i,1) = x(i,1);
    k_positions(i,2) = x(i,2);
  });
  auto positions = torch::from_blob(
    k_positions.data(),
    {n_nodes,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- cell -----
  // TODO: how to use kokkos here?
  auto cell = torch::zeros({3,3}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  cell[0][0] = h0;
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = h5;
  cell[1][1] = h1;
  cell[1][2] = 0.0;
  cell[2][0] = h4;
  cell[2][1] = h3;
  cell[2][2] = h2;

  // ----- edge_index and unit_shifts -----
  // count total number of edges
  auto k_n_edges_vec = Kokkos::View<int64_t*,DeviceType>("k_n_edges_vec", n_nodes);
  Kokkos::parallel_for("PairMACEKokkos: Fill k_n_edges_vec.", n_nodes, KOKKOS_LAMBDA (const int ii) {
    const int i = d_ilist(ii);
    const double xtmp = x(i,0);
    const double ytmp = x(i,1);
    const double ztmp = x(i,2);
    for (int jj=0; jj<d_numneigh(i); ++jj) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const double delx = xtmp - x(j,0);
      const double dely = ytmp - x(j,1);
      const double delz = ztmp - x(j,2);
      const double rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < r_max_squared) {
        k_n_edges_vec(ii) += 1;
      }
    }
  });
  // WARNING: if n_edges remains 0 (e.g., because atoms too far apart)
  // strange things happen on gpu
  int64_t n_edges = 0;
  Kokkos::parallel_reduce("PairMACEKokkos: Determine n_edges.", n_nodes, KOKKOS_LAMBDA(const int ii, int64_t& n_edges) {
    n_edges += k_n_edges_vec(ii);
  }, n_edges);
  // make first_edge vector to help with parallelizing following loop
  auto k_first_edge = Kokkos::View<int64_t*,DeviceType>("k_first_edge", n_nodes);  // initialized to zero
  // TODO: this is serial to avoid race ... is there something better?
  Kokkos::parallel_for("PairMACEKokkos: Fill k_first_edge.", 1, KOKKOS_LAMBDA(const int i) {
    for (int ii=0; ii<n_nodes-1; ++ii) {
      k_first_edge(ii+1) = k_first_edge(ii) + k_n_edges_vec(ii);
    }
  });
  auto k_edge_index = Kokkos::View<int64_t**,Kokkos::LayoutRight,DeviceType>("k_edge_index", 2, n_edges);
  auto k_unit_shifts = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_unit_shifts", n_edges);
  auto k_shifts = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_shifts", n_edges);

  if (domain_decomposition) {

    Kokkos::parallel_for("PairMACEKokkos: Fill edge_index (using domain decomposition).", n_nodes, KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      const double xtmp = x(i,0);
      const double ytmp = x(i,1);
      const double ztmp = x(i,2);
      int k = k_first_edge(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        const double delx = xtmp - x(j,0);
        const double dely = ytmp - x(j,1);
        const double delz = ztmp - x(j,2);
        const double rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < r_max_squared) {
          k_edge_index(0,k) = i;
          k_edge_index(1,k) = j;
          k++;
        }
      }
    });

  } else {

    Kokkos::parallel_for("PairMACEKokkos: Fill edge_index (no domain decomposition).", n_nodes, KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      const double xtmp = x(i,0);
      const double ytmp = x(i,1);
      const double ztmp = x(i,2);
      int k = k_first_edge(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        const double delx = xtmp - x(j,0);
        const double dely = ytmp - x(j,1);
        const double delz = ztmp - x(j,2);
        const double rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < r_max_squared) {
          k_edge_index(0,k) = i;
          int j_local = AtomKokkos::map_kokkos<DeviceType>(tag(j),map_style,k_map_array,k_map_hash);
          k_edge_index(1,k) = j_local;
          double shiftx = x(j,0) - x(j_local,0);
          double shifty = x(j,1) - x(j_local,1);
          double shiftz = x(j,2) - x(j_local,2);
          double shiftxs = std::round(hinv0*shiftx + hinv5*shifty + hinv4*shiftz);
          double shiftys = std::round(hinv1*shifty + hinv3*shiftz);
          double shiftzs = std::round(hinv2*shiftz);
          k_unit_shifts(k,0) = shiftxs;
          k_unit_shifts(k,1) = shiftys;
          k_unit_shifts(k,2) = shiftzs;
          k_shifts(k,0) = h0*shiftxs + h5*shiftys + h4*shiftzs;
          k_shifts(k,1) = h1*shiftys + h3*shiftzs;
          k_shifts(k,2) = h2*shiftzs;
          k++;
        }
      }
    });
  }
  auto edge_index = torch::from_blob(
    k_edge_index.data(),
    {2,n_edges},
    torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto unit_shifts = torch::from_blob(
    k_unit_shifts.data(),
    {n_edges,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto shifts = torch::from_blob(
    k_shifts.data(),
    {n_edges,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- node_attrs -----
  // node_attrs is one-hot encoding for atomic numbers
  int n_node_feats = _mace_atomic_numbers_size;
  auto k_node_attrs = Kokkos::View<double**,Kokkos::LayoutRight,DeviceType>("k_node_attrs", n_nodes, n_node_feats);
  Kokkos::parallel_for("PairMACEKokkos: Fill k_node_attrs.", n_nodes, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    const int lammps_type = type(i);
    int t = -1;
    for (int j=0; j<_mace_atomic_numbers_size; ++j) {
      if (_k_mace_atomic_numbers(j)==_k_lammps_atomic_numbers(lammps_type-1)) {
        t = j+1;
      }
    }
    k_node_attrs(i,t-1) = 1.0;
  });
  auto node_attrs = torch::from_blob(
    k_node_attrs.data(),
    {n_nodes, n_node_feats},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- mask for ghost -----
  Kokkos::View<bool*,DeviceType> k_mask("k_mask", n_nodes);
  Kokkos::parallel_for("PairMACEKokkos: Fill k_mask.", nlocal, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    k_mask(i) = true;
  });
  auto mask = torch::from_blob(
    k_mask.data(),
    n_nodes,
    torch::TensorOptions().dtype(torch::kBool).device(device));

  // TODO: why is batch of size n_nodes?
  auto batch = torch::zeros({n_nodes}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto energy = torch::empty({1}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto forces = torch::empty({n_nodes,3}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto ptr = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto weight = torch::empty({1}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  ptr[0] = 0;
  ptr[1] = n_nodes;
  weight[0] = 1.0;

  // pack the input, call the model, extract the output
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("energy", energy);
  input.insert("forces", forces);
  input.insert("node_attrs", node_attrs);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);
  input.insert("weight", weight);
  auto output = model.forward({input, mask, bool(vflag_global)}).toGenericDict();

  // mace energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    auto node_energy = output.at("node_energy").toTensor();
    auto node_energy_ptr = static_cast<double*>(node_energy.data_ptr());
    auto k_node_energy = Kokkos::View<double*,Kokkos::LayoutRight,DeviceType,Kokkos::MemoryTraits<Kokkos::Unmanaged>>(node_energy_ptr,n_nodes);
    eng_vdwl = 0.0;
    Kokkos::parallel_reduce("PairMACEKokkos: Accumulate site energies.", nlocal, KOKKOS_LAMBDA(const int ii, double &eng_vdwl) {
      const int i = d_ilist(ii);
      eng_vdwl += k_node_energy(i);
    }, eng_vdwl);
  }

  // mace forces
  //   -> derivatives of total mace energy
  forces = output.at("forces").toTensor();
  auto forces_ptr = static_cast<double*>(forces.data_ptr());
  auto k_forces = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType,Kokkos::MemoryTraits<Kokkos::Unmanaged>>(forces_ptr,n_nodes);
  Kokkos::parallel_for("PairMACEKokkos: Extract k_forces.", n_nodes, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    f(i,0) += k_forces(i,0);
    f(i,1) += k_forces(i,1);
    f(i,2) += k_forces(i,2);
  });

  // mace virials (local atoms only)
  //   -> derivatives of sum of site energies of local atoms
  if (vflag_global) {
    // TODO: is this cpu transfer necessary?
    auto vir = output.at("virials").toTensor().to("cpu");
    // caution: lammps does not use voigt ordering
    virial[0] += vir[0][0][0].item<double>();
    virial[1] += vir[0][1][1].item<double>();
    virial[2] += vir[0][2][2].item<double>();
    virial[3] += 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>());
    virial[4] += 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>());
    virial[5] += 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>());
  }

  // TODO: investigate this
  // Appears to be important for dumps and probably more
  atomKK->modified(execution_space,F_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  PairMACE::coeff(narg,arg);

  // new
  k_lammps_atomic_numbers = Kokkos::View<int64_t*,DeviceType>("k_lammps_atomic_numbers",lammps_atomic_numbers.size());
  auto k_lammps_atomic_numbers_mirror = Kokkos::create_mirror_view(k_lammps_atomic_numbers);
  for (int i=0; i<lammps_atomic_numbers.size(); ++i) {
    k_lammps_atomic_numbers_mirror(i) = lammps_atomic_numbers[i];
  }
  Kokkos::deep_copy(k_lammps_atomic_numbers, k_lammps_atomic_numbers_mirror);
  k_mace_atomic_numbers = Kokkos::View<int64_t*,DeviceType>("k_mace_atomic_numbers",mace_atomic_numbers.size());
  auto k_mace_atomic_numbers_mirror = Kokkos::create_mirror_view(k_mace_atomic_numbers);
  for (int i=0; i<mace_atomic_numbers.size(); ++i) {
    k_mace_atomic_numbers_mirror(i) = mace_atomic_numbers[i];
  }
  Kokkos::deep_copy(k_mace_atomic_numbers, k_mace_atomic_numbers_mirror);
  mace_atomic_numbers_size = mace_atomic_numbers.size();
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::init_style()
{
  PairMACE::init_style();
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
}

template<class DeviceType>
double PairMACEKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairMACE::init_one(i,j);
  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  return cutone;
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::allocate()
{
  PairMACE::allocate();
  int n = atom->ntypes + 1;
  MemKK::realloc_kokkos(k_cutsq, "mace:cutsq", n, n);
  d_cutsq = k_cutsq.template view<DeviceType>();
}

namespace LAMMPS_NS {
template class PairMACEKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMACEKokkos<LMPHostType>;
#endif
}

