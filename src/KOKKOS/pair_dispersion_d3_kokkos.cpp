#include "pair_dispersion_d3_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include <cmath>
#include "error.h"
#include "comm.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "update.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h" 
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

using namespace LAMMPS_NS;

//static constexpr double K1 = 16.0;
//static constexpr double K3 = -4.0;
static constexpr int NUM_ELEMENTS=94;
static constexpr int N_PARS_COLS=5;  // number columns in C6 table
static constexpr int N_PARS_ROWS=32395; // number of rows C6 table
//static constexpr double autoang =  0.52917725 ; // au-Bohr to angstroms
//static constexpr double autoev  = 27.21140795 ; // au-Hartree to eV

#include <d3_parameters.h>
//#define KOKKOS_LAMBDA [=, this] 

//template<class DeviceType>
//class PairDispersionD3Kokkos;

/*
struct params_d3
{
      F_FLOAT cut;      // cutoff distance
      F_FLOAT cutsq;    // squared cutoff
      F_FLOAT s6;       // scaling factor for C6 term
      F_FLOAT s8;       // scaling factor for C8 term (if used)
      F_FLOAT rs6;      // damping range scaling for C6
      F_FLOAT rs8;      // damping range scaling for C8
      F_FLOAT alpha;    // damping exponent
      F_FLOAT r0_ab;    // reference R0 for pair (i,j)
      F_FLOAT pad[1];
};
*/
template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp)
:PairDispersionD3(lmp)
{
  comm_forward = 2;
  comm_reverse = 2;
  
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom ; 
//  Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_params;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
  damping_type = "zero_damping";
}




struct DC6 
{
  double C6;
  double dC6i;
  double dC6j;

  KOKKOS_INLINE_FUNCTION
  DC6() : C6(0), dC6i(0), dC6j(0) {}
};

template<class DeviceType, class MXCI_View, class C6AB_View>
KOKKOS_INLINE_FUNCTION
DC6 get_dC6_res(
    int iat, int jat,
    double cni, double cnj,
    const MXCI_View  &d_mxci,
    const C6AB_View  &d_c6ab)
{
  //const inline constexpr double      K3 = -4.0;
  //const inline constexpr double      K1 = 16.0;
  //const inline constexpr double autoang =  0.52917725;
  //const inline constexpr double autoev  = 27.21140795;
  DC6 out;

  double c6_ref, cni_ref, cnj_ref;
  double c6mem  = -1.0e20;
  double r_save =  1.0e20;
  double r, expterm, term;
  double num = 0.0, den = 0.0;
  double KKnum_i = 0.0, KKnum_j = 0.0;
  double KKden_i = 0.0, KKden_j = 0.0;

  for (int ci = 0; ci <= d_mxci(iat); ci++) {
    for (int cj = 0; cj <= d_mxci(jat); cj++) {
      c6_ref = d_c6ab(iat, jat, ci, cj, 0) * autoev * pow(autoang, 6);

      if (c6_ref > 0.0) {
        cni_ref = d_c6ab(iat, jat, ci, cj, 1);
        cnj_ref = d_c6ab(iat, jat, ci, cj, 2);

        r = (cni - cni_ref) * (cni - cni_ref) +
            (cnj - cnj_ref) * (cnj - cnj_ref);

        if (r < r_save) { r_save = r; c6mem = c6_ref; }

        expterm = exp(K3 * r);
        num += c6_ref * expterm;
        den += expterm;

        expterm = expterm * 2.0 * K3;

        term     = expterm * (cni - cni_ref);
        KKnum_i += c6_ref * term;
        KKden_i += term;

        term     = expterm * (cnj - cnj_ref);
        KKnum_j += c6_ref * term;
        KKden_j += term;
      }
    }
  }

  if (den > 1.0E-99) {
    out.C6   = num / den;
    out.dC6i = ((KKnum_i * den) - (KKden_i * num)) / (den * den);
    out.dC6j = ((KKnum_j * den) - (KKden_j * num)) / (den * den);
  } else {
    out.C6   = c6mem;
    out.dC6i = 0.0;
    out.dC6j = 0.0;
  }

  return out;
}
template <class DualViewType, class ExecSpace, class FillerFunc>
inline void init_dualview_from_host(DualViewType &dv, const FillerFunc &filler)
{
  auto hv = dv.h_view;
  using HostView = std::decay_t<decltype(hv)>;
  constexpr int rank = HostView::rank;

  if constexpr (rank == 1) {
    for (size_t i = 0; i < hv.extent(0); ++i) filler(hv, static_cast<int>(i));
  } else if constexpr (rank == 2) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        filler(hv, static_cast<int>(i), static_cast<int>(j));
  } else if constexpr (rank == 3) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k));
  } else if constexpr (rank == 4) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          for (size_t l = 0; l < hv.extent(3); ++l)
            filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(l));
  } else if constexpr (rank == 5) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          for (size_t l = 0; l < hv.extent(3); ++l)
            for (size_t m = 0; m < hv.extent(4); ++m)
              filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k),
                         static_cast<int>(l), static_cast<int>(m));
  } else {
    static_assert(rank <= 5, "Rank > 5 not implemented");
  }

  dv.modify_host();
  dv.template sync<ExecSpace>();
}

template<class DeviceType>
struct PairDispD3Kernel_dEdIJ {
  using exec_space      = typename DeviceType::execution_space;
  using View2D_F        = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using View1D_F        = Kokkos::View<F_FLOAT*, DeviceType>;
  using View1D_int      = Kokkos::View<int*, DeviceType>;
  using View2D_params   = Kokkos::View<params_d3**, Kokkos::LayoutRight, DeviceType>;
  using View5D_c6ab     = Kokkos::View<double*****, Kokkos::LayoutRight, DeviceType>;

  // views
  View2D_F      d_x;
  View2D_F      d_f;
  View1D_F      d_cn_v;
  View1D_F      d_dc6_v;
  View1D_int    d_type;
  View1D_int      d_special_lj;
  View1D_int    d_mxci;
  View5D_c6ab   d_c6ab;
  View2D_params d_params;
  View1D_F      d_r2r4v;
  View1D_int    d_ilist_v;
  View1D_int    d_numneigh;

  // neighbor list access
  NeighListKokkos<DeviceType>* k_list;

  // owner (for ev_tally on device)
  PairDispersionD3Kokkos<DeviceType>* self;

  // scalars
  bool    l_eflag;
  bool    l_newton_pair;
  int     l_nlocal;
  F_FLOAT autoang;

  PairDispD3Kernel_dEdIJ(
    View2D_F d_x_,
    View2D_F d_f_,
    View1D_F d_cn_v_,
    View1D_F d_dc6_v_,
    View1D_int d_type_,
    View1D_int d_special_lj_,
    View1D_int d_mxci_,
    View5D_c6ab d_c6ab_,
    View2D_params d_params_,
    View1D_F d_r2r4v_,
    View1D_int d_ilist_v_,
    View1D_int d_numneigh_,
    NeighListKokkos<DeviceType>* k_list_,
    PairDispersionD3Kokkos<DeviceType>* self_,
    bool l_eflag_,
    bool l_newton_pair_,
    int  l_nlocal_,
    F_FLOAT autoang_)
  : d_x(d_x_), d_f(d_f_), d_cn_v(d_cn_v_), d_dc6_v(d_dc6_v_), d_type(d_type_),
    d_special_lj(d_special_lj_), d_mxci(d_mxci_), d_c6ab(d_c6ab_), d_params(d_params_),
    d_r2r4v(d_r2r4v_), d_ilist_v(d_ilist_v_), d_numneigh(d_numneigh_),
    k_list(k_list_), self(self_), l_eflag(l_eflag_), l_newton_pair(l_newton_pair_),
    l_nlocal(l_nlocal_), autoang(autoang_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    const int   i     = d_ilist_v(ii);
    const int   itype = d_type(i);
    const auto  icn   = d_cn_v(i);
    const int   jnum  = d_numneigh(i);
    const auto  neigh_i = k_list->get_neighbors(i);

    for (int jj = 0; jj < jnum; jj++) {
      const int jenc   = neigh_i(jj);
      const int j      = jenc & NEIGHMASK;
      const int sbmask = jenc >> SBBITS;
      const int jtype  = d_type(j);
      const auto jcn   = d_cn_v(j);
      const F_FLOAT factor = d_special_lj(sbmask);

      const F_FLOAT delx = d_x(i,0) - d_x(j,0);
      const F_FLOAT dely = d_x(i,1) - d_x(j,1);
      const F_FLOAT delz = d_x(i,2) - d_x(j,2);
      const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

      const auto &p = d_params(itype, jtype);
      if (rsq < p.cutsq) {
        const DC6 t_dc6_res = get_dC6_res<DeviceType>(itype, jtype, icn, jcn, d_mxci, d_c6ab);
        const F_FLOAT C6 = t_dc6_res.C6;
        const F_FLOAT C8 = F_FLOAT(3.0) * C6 * d_r2r4v(itype) * d_r2r4v(jtype) * autoang * autoang;

        const F_FLOAT r        = sqrt(rsq);
        const F_FLOAT r2inv    = F_FLOAT(1.0) / rsq;
        const F_FLOAT r6inv    = r2inv * r2inv * r2inv;
        const F_FLOAT r8inv    = r6inv * r2inv;

        const F_FLOAT r0       = r / p.r0_ab;
        const F_FLOAT alpha6   = p.alpha;
        const F_FLOAT alpha8   = alpha6 + F_FLOAT(2.0);

        const F_FLOAT t6    = pow(p.rs6 / r0, alpha6);
        const F_FLOAT damp6 = F_FLOAT(1.0) / (F_FLOAT(1.0) + F_FLOAT(6.0) * t6);

        const F_FLOAT t8    = pow(p.rs8 / r0, alpha8);
        const F_FLOAT damp8 = F_FLOAT(1.0) / (F_FLOAT(1.0) + F_FLOAT(6.0) * t8);

        const F_FLOAT e6    = C6 * damp6 * r6inv;
        const F_FLOAT e8    = C8 * damp8 * r8inv;

        const F_FLOAT tmp6  = F_FLOAT(6.0) * p.s6 * C6 * r2inv * r6inv * damp6;
        const F_FLOAT tmp8  = F_FLOAT(8.0) * p.s8 * C8 * r2inv * r8inv * damp8;

        const F_FLOAT fpair1 = -tmp6 - tmp8;
        const F_FLOAT fpair2 = tmp6 * alpha6 * t6 * damp6 + (F_FLOAT(3.0)/F_FLOAT(4.0)) * tmp8 * alpha8 * t8 * damp8;
        const F_FLOAT fpair  = (fpair1 + fpair2) * factor;

        // energy tally (local)
        F_FLOAT evdwl_local = F_FLOAT(0.0);
        if (l_eflag) evdwl_local = -(p.s6 * e6 + p.s8 * e8) * factor;

        const F_FLOAT rest = (p.s6 * e6 + p.s8 * e8) / C6;

        Kokkos::atomic_add(&d_dc6_v(i), rest * t_dc6_res.dC6i);
        if (l_newton_pair || j < l_nlocal)
          Kokkos::atomic_add(&d_dc6_v(j), rest * t_dc6_res.dC6j);

        // forces
        const F_FLOAT fx = fpair * delx;
        const F_FLOAT fy = fpair * dely;
        const F_FLOAT fz = fpair * delz;

        d_f(i,0) += fx;  d_f(i,1) += fy;  d_f(i,2) += fz;
        Kokkos::atomic_add(&d_f(j,0), -fx);
        Kokkos::atomic_add(&d_f(j,1), -fy);
        Kokkos::atomic_add(&d_f(j,2), -fz);

        if (l_eflag) {
          self->ev_tally(i, j, l_nlocal, l_newton_pair, evdwl_local, F_FLOAT(0.0), fpair, delx, dely, delz);
        }
      }
    }
  }
};

template<class DeviceType>
struct PairDispD3Kernel_dEdXYZ {
  using exec_space      = typename DeviceType::execution_space;
  using View2D_F        = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using View1D_F        = Kokkos::View<F_FLOAT*, DeviceType>;
  using View1D_int      = Kokkos::View<int*, DeviceType>;
  using View2D_params   = Kokkos::View<params_d3**, Kokkos::LayoutRight, DeviceType>;

  // views
  View2D_F      d_x;
  View2D_F      d_f;
  View1D_F      d_dc6_v;
  View1D_int    d_type;
  View1D_int    d_special_lj;
  View1D_F      d_rcov;
  View2D_params d_params;   // to reuse per-type cutoffs

  // neighbor list
  View1D_int    d_ilist_v;
  View1D_int    d_numneigh;
  NeighListKokkos<DeviceType>* k_list;

  // owner for ev_tally
  PairDispersionD3Kokkos<DeviceType>* self;

  // scalars
  bool    l_newton_pair;
  bool    l_evflag;
  int     l_nlocal;
  F_FLOAT autoang;
  F_FLOAT d_cn_thr;
  F_FLOAT K1;

  PairDispD3Kernel_dEdXYZ(
    View2D_F d_x_,
    View2D_F d_f_,
    View1D_F d_dc6_v_,
    View1D_int d_type_,
    View1D_int d_special_lj_,
    View1D_F d_rcov_,
    View2D_params d_params_,
    View1D_int d_ilist_v_,
    View1D_int d_numneigh_,
    NeighListKokkos<DeviceType>* k_list_,
    PairDispersionD3Kokkos<DeviceType>* self_,
    bool l_newton_pair_,
    bool l_evflag_,
    int  l_nlocal_,
    F_FLOAT autoang_,
    F_FLOAT d_cn_thr_,
    F_FLOAT K1_)
  : d_x(d_x_), d_f(d_f_), d_dc6_v(d_dc6_v_), d_type(d_type_), d_special_lj(d_special_lj_),
    d_rcov(d_rcov_), d_params(d_params_), d_ilist_v(d_ilist_v_), d_numneigh(d_numneigh_),
    k_list(k_list_), self(self_), l_newton_pair(l_newton_pair_), l_evflag(l_evflag_),
    l_nlocal(l_nlocal_), autoang(autoang_), d_cn_thr(d_cn_thr_), K1(K1_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    const int i     = d_ilist_v(ii);
    const int itype = d_type(i);
    const int jnum  = d_numneigh(i);
    const auto neigh_i   = k_list->get_neighbors(i);
    const F_FLOAT rcov_i = d_rcov(itype);

    for (int jj = 0; jj < jnum; jj++) {
      const int jenc   = neigh_i(jj);
      const int j      = jenc & NEIGHMASK;
      const int sbmask = jenc >> SBBITS;
      const int jtype  = d_type(j);
      const F_FLOAT rcov_j = d_rcov(jtype);
      const F_FLOAT factor = d_special_lj(sbmask);

      const F_FLOAT delx = d_x(i,0) - d_x(j,0);
      const F_FLOAT dely = d_x(i,1) - d_x(j,1);
      const F_FLOAT delz = d_x(i,2) - d_x(j,2);
      const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

      const auto &p = d_params(itype, jtype); // use same per-type cutoffs
      if (rsq < p.cutsq) {
        F_FLOAT dcn = F_FLOAT(0.0);
        if (rsq < d_cn_thr) {
          const F_FLOAT rcovij  = (rcov_i + rcov_j) * autoang;
          const F_FLOAT invr    = F_FLOAT(1.0) / sqrt(rsq);
          const F_FLOAT expterm = exp(-K1 * (rcovij * invr - F_FLOAT(1.0)));
          const F_FLOAT denom   = (expterm + F_FLOAT(1.0));
          dcn = -K1 * rcovij * expterm / (rsq * denom * denom);
        }

        const F_FLOAT invr  = F_FLOAT(1.0) / sqrt(rsq);
        const F_FLOAT fpair = dcn * (d_dc6_v(i) + d_dc6_v(j)) * invr * factor;

        const F_FLOAT fx = fpair * delx;
        const F_FLOAT fy = fpair * dely;
        const F_FLOAT fz = fpair * delz;

        d_f(i,0) += fx; d_f(i,1) += fy; d_f(i,2) += fz;

        if (l_newton_pair || j < l_nlocal) {
          Kokkos::atomic_add(&d_f(j,0), -fx);
          Kokkos::atomic_add(&d_f(j,1), -fy);
          Kokkos::atomic_add(&d_f(j,2), -fz);
        }

        if (l_evflag) {
          self->ev_tally(i, j, l_nlocal, l_newton_pair, F_FLOAT(0.0), F_FLOAT(0.0), fpair, delx, dely, delz);
        }
      }
    }
  }
};




/*
template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp)
:PairDispersionD3(lmp)
{
  comm_forward = 2;
  comm_reverse = 2;
  
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom ; 
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
  damping_type = "zero_damping";
}
*/

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  if (copymode) return;
  if (allocated) 
  {
    //memoryKK->destroy_kokkos(k_eatom, eatom);
    //memoryKK->destroy_kokkos(k_vatom, vatom);
    //memoryKK->destroy_kokkos(k_cutsq, cutsq); 
    //memoryKK->destroy_kokkos(k_r0ab, r0ab);
    //memoryKK->destroy_kokkos(k_c6ab, c6ab);
    memoryKK->destroy_kokkos(k_r2r4, r2r4);
    memoryKK->destroy_kokkos(k_rcov, rcov); 
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::allocateKK()
{
  //PairDispersionD3::allocate();
  
  int n = atom->ntypes;
  
  // 1D views 
  memory->destroy(mxci);
  memory->destroy(r2r4);
  memory->destroy(rcov);
  memoryKK->create_kokkos(k_mxci, mxci, n+1, "pair:mxci");
  memoryKK->create_kokkos(k_r2r4, r2r4, n+1, "pair:r2r4");
  memoryKK->create_kokkos(k_rcov, rcov, n+1, "pair:rcov");
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::settings(int narg, char **arg)
{
  //PairDispersionD3::settings(int narg, char **arg);
  PairDispersionD3::settings(narg, arg); 
  k_rthr = this->rthr;
  k_cn_thr = this->cn_thr;
  //Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_params;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  // Call base class 
  PairDispersionD3::coeff(narg, arg);
  allocateKK();
}

template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j) {
  double cut = PairDispersionD3::init_one(i, j);
  // Optionally capture cutsq for this pair here into k_params.h_view(i,j).cutsq
  return cut;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style() {
  PairDispersionD3::init_style();
  sync_coeffs_to_device();
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                         !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
   
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::sync_coeffs_to_device()
{
  const int ntypes = atom->ntypes;
  
  // Allocate params + 1D r2r4
  k_params = Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType>("d3_params", ntypes+1, ntypes+1);
  k_r2r4   = Kokkos::DualView<F_FLOAT*, DeviceType>("r2r4", ntypes+1); 
  k_mxci   = Kokkos::DualView<int*, DeviceType>("t_mxci", ntypes+1); 
  k_c6ab   = Kokkos::DualView<double*****, Kokkos::LayoutRight, DeviceType>("c6ab", ntypes+1, ntypes+1, 5, 5, 3);
  
  // ---- fill params on host ----
  {
    auto hv = k_params.h_view;
    for (int i = 1; i <= ntypes; i++) {
      k_r2r4.h_view(i) = r2r4[i];
      for (int j = 1; j <= ntypes; j++) {
        auto &p = hv(i,j);
        p.cutsq = cutsq[i][j];
        p.cut   = std::sqrt(p.cutsq);
        p.s6    = s6;
        p.s8    = s8;
        p.rs6   = rs6;
        p.rs8   = rs8;
        p.alpha = alpha;
        p.r0_ab = r0ab[i][j];
      }
    }
  }
  k_params.modify_host();
  k_params.template sync<DeviceType>();

  // ---- r2r4 via filler ----
  auto fill_r2r4 = [&](auto &hv, int i) { hv(i) = r2r4[i]; };
  init_dualview_from_host<decltype(k_r2r4), typename DeviceType::execution_space>(k_r2r4, fill_r2r4);
  

  // ---- mxci + c6ab ----
  int gi_max = 0, gj_max = 0;
  for (int t = 1; t <= ntypes; t++) {
    gi_max = std::max(gi_max, mxci[t]);
    gj_max = std::max(gj_max, mxci[t]);
  }
  gi_max++; gj_max++;

  init_dualview_from_host<decltype(k_mxci), typename DeviceType::execution_space>(k_mxci, [&](auto &hv, int i) { hv(i) = mxci[i]; });

  
  init_dualview_from_host<decltype(k_c6ab), typename DeviceType::execution_space>(k_c6ab, [&](auto &hv, int i, int j, int gi, int gj, int k) 
  {
    if (gi <= mxci[i] && gj <= mxci[j]) hv(i,j,gi,gj,k) = c6ab[i][j][gi][gj][k];
  });


 
  /*
  init_dualview_from_host<decltype(k_c6ab), decltype([&](auto&,int,int,int,int,int){}), DeviceType>
    (k_c6ab, [&](auto &hv, int i, int j, int gi, int gj, int k) {
      if (gi <= mxci[i] && gj <= mxci[j])
        hv(i,j,gi,gj,k) = c6ab[i][j][gi][gj][k];
    }); */ 
}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_numberKK() {
  // 1) Sync atoms needed
  atomKK->sync(execution_space, datamask_read);

  // 2) Grow per-atom buffers to capacity before taking views
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    k_cn  = typename AT::tdual_double_1d("pair:cn",  nmax);
    k_dc6 = typename AT::tdual_double_1d("pair:dc6", nmax);
  }

  // 3) Get device neighbor list Views/handles
  auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_ilist     = k_list->d_ilist;        // View<int* , ... , Device>
  auto d_numneigh  = k_list->d_numneigh;     // View<int* , ... , Device>
  //auto d_firstneigh = k_list->d_firstneigh;  // View<int** , ... , Device> (2-D neighbors)

  // Optional fail-fast checks
  if (!d_ilist.data() || !d_numneigh.data()) {
    error->all(FLERR, "Device neighbor list not available");
  }

  // 4) Take device views of your arrays
  auto d_cn   = k_cn.template view<DeviceType>();
  auto d_dc6  = k_dc6.template view<DeviceType>();
  auto d_x    = k_x.template view<DeviceType>();
  auto d_rcov = k_rcov.template view<DeviceType>();   // must be sized to ntypes+1
  auto d_type = k_type.template view<DeviceType>();   // sized to nmax

  const int nlocal      = atom->nlocal;
  const int nall        = nlocal + atom->nghost;
  const int newton_pair = force->newton_pair;
  const int inum        = list->inum;

  // 5) Zero
  Kokkos::parallel_for(
    "ZeroCN_DC6",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, newton_pair ? nall : nlocal),
    KOKKOS_LAMBDA(const int i) {
      d_cn(i)  = F_FLOAT(0);
      d_dc6(i) = F_FLOAT(0);
    });

  // 6) Compute using only device Views
  const F_FLOAT d_cn_thr = static_cast<F_FLOAT>(cn_thr);
  Kokkos::parallel_for(
    "PairDispersionD3KokkosCalcCNKK",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
    KOKKOS_LAMBDA(const int ii) {
      const int i     = d_ilist(ii);
      const int itype = d_type(i);
      const int jnum  = d_numneigh(i);
      const auto rcov_i = d_rcov(itype);

      for (int jj = 0; jj < jnum; ++jj) {
        int jenc = d_firstneigh(i, jj);
        const int j = jenc & NEIGHMASK;

        const int jtype = d_type(j);
        const auto rcov_j = d_rcov(jtype);

        const F_FLOAT delx = d_x(i,0) - d_x(j,0);
        const F_FLOAT dely = d_x(i,1) - d_x(j,1);
        const F_FLOAT delz = d_x(i,2) - d_x(j,2);
        const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

        if (rsq > d_cn_thr) continue;

        const F_FLOAT rr      = sqrt(rsq);
        const F_FLOAT rcov_ij = (rcov_i + rcov_j) * autoang;
        const F_FLOAT cn_ij   = F_FLOAT(1) / (F_FLOAT(1) + exp(-K1 * ((rcov_ij/rr) - F_FLOAT(1))));

        d_cn(i) += cn_ij;
        if (newton_pair || j < nlocal) Kokkos::atomic_add(&d_cn(j), cn_ij);
      }
    });

  // 7) Comm as before
  communicationStage = 1;
  if (newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);
}

/*

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_numberKK()
{
  // sync 
  atomKK->sync(execution_space, datamask_read);
  
  auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    k_cn  = typename  AT::tdual_double_1d("pair:cn",  nmax);
    k_dc6 = typename AT::tdual_double_1d("pair:dc6", nmax);
  }
  

 
  //NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_ilist_v       = k_list->d_ilist;
  auto d_numneigh    = k_list->d_numneigh;
  //auto k_firstneigh  = k_list->d_firstneigh;
  //const F_FLOAT d_cn_thr   = this->k_cn_thr;
  const F_FLOAT d_cn_thr = static_cast<F_FLOAT>(this->cn_thr);
  
  int nlocal      = atom->nlocal;
  int nall        = nlocal + atom->nghost;
  int newton_pair = force->newton_pair; 
  
  auto d_cn_v         = k_cn.template view<DeviceType>();
  auto d_dc6_v        = k_dc6.template view<DeviceType>(); 
  //auto d_ilist_v      = k_ilist.template view<DeviceType>();
  //auto d_numneigh   = k_numneigh.template view<DeviceType>();
  //auto d_firstneigh = k_firstneigh.template view<DeviceType>();
  auto d_x          = k_x.template view<DeviceType>();
  auto d_rcov       = k_rcov.template view<DeviceType>();
  auto d_type       = k_type.template view<DeviceType>(); 
   
  
  const int inum  = list->inum;
  
  Kokkos::parallel_for("ZeroCN_DC6", Kokkos::RangePolicy
                       <typename DeviceType::execution_space>(0, newton_pair ? nall : nlocal),
  KOKKOS_LAMBDA (const int i) 
  {
    d_cn_v(i)  = F_FLOAT(0);
    d_dc6_v(i) = F_FLOAT(0);
  });
 
  Kokkos::parallel_for("PairDispersionD3KokkosCalcCNKK",
  Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
  KOKKOS_LAMBDA ( const int ii ) 
  {
    const int i                     = d_ilist_v(ii);
    const int itype                 = d_type(i);
    const int jnum                  = d_numneigh(i); 
    //const auto * __restrict__ jlist = d_firstneigh(i);
    // replace d_firstneigh aka jlist with neigh_i
    const auto neigh_i              = k_list->get_neighbors(i);
    const auto d_rcov_i              = d_rcov(itype);
    for (int jj = 0; jj < jnum; jj++)
    {
      const int jenc                = neigh_i(jj);
      const int j                   = jenc & NEIGHMASK;
      const int jtype               = d_type(j);  
      const auto d_rcov_j           = d_rcov(jtype); 
      const F_FLOAT delx            = d_x(i, 0) - d_x(j, 0);
      const F_FLOAT dely            = d_x(i, 1) - d_x(j, 1); 
      const F_FLOAT delz            = d_x(i, 2) - d_x(j, 2); 
      
      const F_FLOAT rsq             = delx*delx + dely*dely + delz*delz; 
   
      if (rsq > d_cn_thr) continue;
     
      const F_FLOAT rr              = sqrt(rsq);
      const F_FLOAT rcov_ij         = (d_rcov_i + d_rcov_j) * autoang; 
      const F_FLOAT cn_ij           = F_FLOAT(1.0) / ( F_FLOAT(1.0) + exp(-K1 * ((rcov_ij/rr) - F_FLOAT(1.0)))); 

      d_cn_v(i) += cn_ij ; 
      if (newton_pair || j < nlocal) {Kokkos::atomic_add(&d_cn_v(j), cn_ij);} 
    }
  });
 
  communicationStage = 1;
  if (newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this); 
}
*/
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag, int vflag)
{
  // init energy/virial flags
  ev_init(eflag, vflag);

  // 1) coordination numbers (fills k_cn and zeros k_dc6)
  calc_coordination_numberKK();

  // common flags and sizes
  const bool l_eflag       = static_cast<bool>(eflag);
  const bool l_evflag      = static_cast<bool>(evflag);
  const bool l_newton_pair = static_cast<bool>(force->newton_pair);
  const int  l_nlocal      = atom->nlocal;
  const int  inum          = list->inum;

  // neighbor list
  auto* k_list      = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto  d_ilist_v   = k_list->d_ilist;
  auto  d_numneigh  = k_list->d_numneigh;

  // device views
  auto d_x          = k_x.template view<DeviceType>();
  auto d_f          = k_f.template view<DeviceType>();
  auto d_type       = k_type.template view<DeviceType>();
  auto d_special_lj = k_special_lj.template view<DeviceType>();

  auto d_cn_v       = k_cn.template view<DeviceType>();
  auto d_dc6_v      = k_dc6.template view<DeviceType>();

  auto d_params     = k_params.template view<DeviceType>();
  auto d_r2r4       = k_r2r4.template view<DeviceType>();
  auto d_rcov       = k_rcov.template view<DeviceType>();

  auto d_mxci       = k_mxci.template view<DeviceType>();
  auto d_c6ab_view  = k_c6ab.template view<DeviceType>();

  // constants used on device
  const F_FLOAT autoang_loc  = static_cast<F_FLOAT>(autoang);
  const F_FLOAT d_cn_thr_val = static_cast<F_FLOAT>(cn_thr);  // squared distance threshold
  const F_FLOAT K1_loc       = static_cast<F_FLOAT>(K1);

  // 2) main pair interaction kernel: forces + energy + dC6 accumulation
  {
    PairDispD3Kernel_dEdIJ<DeviceType> f1(
      d_x, d_f, d_cn_v, d_dc6_v, d_type, d_special_lj, d_mxci, d_c6ab_view,
      d_params, d_r2r4, d_ilist_v, d_numneigh, k_list, this,
      l_eflag, l_newton_pair, l_nlocal, autoang_loc);

    Kokkos::parallel_for(
      "PairDispD3Kokkos_dEdIJ",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
      f1);

    // mark modified on device
    k_f.template modify<DeviceType>();
    k_dc6.template modify<DeviceType>();
  }

  // 3) communicate dC6 (same pattern as the CPU version)
  communicationStage = 2;
  if (l_newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);

  // 4) gradient from dCN/dr using accumulated dC6 (adds to forces)
  {
    PairDispD3Kernel_dEdXYZ<DeviceType> f2(
      d_x, d_f, d_dc6_v, d_type, d_special_lj, d_rcov, d_params,
      d_ilist_v, d_numneigh, k_list, this,
      l_newton_pair, l_evflag, l_nlocal, autoang_loc, d_cn_thr_val, K1_loc);

    Kokkos::parallel_for(
      "PairDispD3Kokkos_dEdXYZ",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
      f2);

    // mark modified on device
    k_f.template modify<DeviceType>();
  }

  // optional virial via f · r
  if (vflag_fdotr) virial_fdotr_compute();
}


/*
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag, int vflag)
{
  // Host-side setup (unchanged)
  atomKK->sync(execution_space, datamask_read);
  ev_init(eflag, vflag);
  calc_coordination_numberKK();

  const bool l_eflag       = eflag;
  const bool l_newton_pair = force->newton_pair;
  const int  l_nlocal      = atom->nlocal;
  const int  inum          = list->inum;
  const bool l_evflag      = evflag;

  auto *k_list = static_cast<NeighListKokkos<DeviceType>*>(list);

  // Sync input DualViews to device
  k_params.template sync<DeviceType>();
  k_r2r4.template  sync<DeviceType>();
  k_cn.template    sync<DeviceType>();

  // Device views
  auto d_x          = k_x.template view<DeviceType>();
  auto d_f          = k_f.template view<DeviceType>();
  auto d_cn_v       = k_cn.template view<DeviceType>();
  auto d_dc6_v      = k_dc6.template view<DeviceType>();
  auto d_type       = k_type.template view<DeviceType>();
  auto d_special_lj = k_special_lj.template view<DeviceType>();
  auto d_mxci       = k_mxci.template view<DeviceType>();
  auto d_c6ab       = k_c6ab.template view<DeviceType>();
  auto d_params     = k_params.template view<DeviceType>();
  auto d_r2r4v      = k_r2r4.template view<DeviceType>();
  auto d_rcov       = k_rcov.template view<DeviceType>();

  auto d_ilist_v    = k_list->d_ilist;
  auto d_numneigh   = k_list->d_numneigh;
  auto d_neighbors  = k_list->d_neighbors;

  const F_FLOAT d_cn_thr_val = static_cast<F_FLOAT>(this->k_cn_thr);

  // ======================
  // Kernel 1: pair forces
  // ======================
  PairDispD3Kernel_dEdIJ<DeviceType> functor1{
    d_x, d_f, d_cn_v, d_dc6_v, d_type,
    d_special_lj, d_mxci, d_c6ab,
    d_params, d_r2r4v,
    d_ilist_v, d_numneigh, d_neighbors,
    k_list, l_eflag, l_newton_pair, l_nlocal,
    autoang
  };

  Kokkos::parallel_for(
    "PairDispD3KokkosComputeKKdEdIJ",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
    functor1
  );

  k_f.template modify<DeviceType>();

  // Communication stage 2
  this->communicationStage = 2;
  if (l_newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);

  // =========================
  // Kernel 2: dCN/dr gradient
  // =========================
  PairDispD3Kernel_dEdXYZ<DeviceType> functor2{
    d_x, d_f, d_dc6_v, d_type, d_special_lj, d_rcov,
    d_ilist_v, d_numneigh, d_neighbors, d_params,
    k_list, l_newton_pair, l_evflag, l_nlocal,
    autoang, d_cn_thr_val, K1
  };

  Kokkos::parallel_for(
    "PairDispD3KokkosComputeKKdEdXYZ",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
    functor2
  );

  if (vflag_fdotr) virial_fdotr_compute();
  k_f.template modify<DeviceType>();
}




*/


// pack_forward_comm
template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm(
    int n,
    DAT::tdual_int_1d k_list,
    DAT::tdual_xfloat_1d &k_buf,
    int /*pbc_flag*/,
    int /*pbc*/
) {
  auto d_list = k_list.view_device();
  auto d_buf  = k_buf.view_device();

  auto d_cn   = k_cn.view_device();
  auto d_dc6  = k_dc6.view_device();

  const int commStage = communicationStage;

  Kokkos::parallel_for(
      "pack_fwd",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int idx = d_list(i);
        d_buf(i) = (commStage == 1) ? d_cn(idx) : d_dc6(idx);
      });

  // Make device writes visible to host for MPI
  k_buf.template modify<DeviceType>();
  k_buf.template sync<LMPHostType>();
  Kokkos::fence();
  return n;
}

// unpack_forward_comm
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm(
    int n,
    int first,
    DAT::tdual_xfloat_1d &k_buf
) {
  auto d_buf = k_buf.view_device();
  auto d_cn  = k_cn.view_device();
  auto d_dc6 = k_dc6.view_device();

  const int commStage = communicationStage;

  Kokkos::parallel_for(
      "unpack_fwd",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int idx = first + i;
        const auto v  = d_buf(i);
        if (commStage == 1) d_cn(idx)  = v;
        else                d_dc6(idx) = v;
      });
}

// pack_reverse_comm
template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm(
    int n,
    int first,
    DAT::tdual_xfloat_1d &k_buf
) {
  auto d_buf = k_buf.view_device();
  auto d_cn  = k_cn.view_device();
  auto d_dc6 = k_dc6.view_device();

  const int commStage = communicationStage;

  Kokkos::parallel_for(
      "pack_rev",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int idx = first + i;
        d_buf(i) = (commStage == 1) ? d_cn(idx) : d_dc6(idx);
      });

  k_buf.template modify<DeviceType>();
  k_buf.template sync<LMPHostType>();
  Kokkos::fence();
  return n;
}

// unpack_reverse_comm
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm(
    int n,
    DAT::tdual_int_1d k_list,
    DAT::tdual_xfloat_1d &k_buf
) {
  auto d_list = k_list.view_device();
  auto d_buf  = k_buf.view_device();
  auto d_cn   = k_cn.view_device();
  auto d_dc6  = k_dc6.view_device();

  const int commStage = communicationStage;

  Kokkos::parallel_for(
      "unpack_rev",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int idx = d_list(i);
        const auto v  = d_buf(i);
        if (commStage == 1) Kokkos::atomic_add(&d_cn(idx),  v);
        else                Kokkos::atomic_add(&d_dc6(idx), v);
      });
}


template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm(
    int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  // simplest: host mirror copy
  int m = 0;
  if (communicationStage == 1) {
    k_cn.sync_host();
    for (int i = 0; i < n; i++) buf[m++] = k_cn.h_view(list[i]);
  } else {
    k_dc6.sync_host();
    for (int i = 0; i < n; i++) buf[m++] = k_dc6.h_view(list[i]);
  }
  return m;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm(
    int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  if (communicationStage == 1) {
    for (int i = first; i < last; i++) k_cn.h_view(i)  = buf[m++];
    k_cn.modify_host();
  } else {
    for (int i = first; i < last; i++) k_dc6.h_view(i) = buf[m++];
    k_dc6.modify_host();
  }
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm(
    int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  if (communicationStage == 1) {
    k_cn.sync_host();
    for (int i = first; i < last; i++) buf[m++] = k_cn.h_view(i);
  } else {
    k_dc6.sync_host();
    for (int i = first; i < last; i++) buf[m++] = k_dc6.h_view(i);
  }
  return m;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm(
    int n, int *list, double *buf)
{
  int m = 0;
  if (communicationStage == 1) {
    for (int i = 0; i < n; i++) k_cn.h_view(list[i]) += buf[m++];
    k_cn.modify_host();
  } else {
    for (int i = 0; i < n; i++) k_dc6.h_view(list[i]) += buf[m++];
    k_dc6.modify_host();
  }
}


namespace LAMMPS_NS {
#ifdef KOKKOS_ENABLE_CUDA
template class PairDispersionD3Kokkos<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template class PairDispersionD3Kokkos<Kokkos::Serial>;
#endif
}


