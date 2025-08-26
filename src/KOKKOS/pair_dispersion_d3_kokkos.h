#ifdef PAIR_CLASS
//clang-format off
PairStyle(dispersion/d3/kk, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/device, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/host, PairDispersionD3Kokkos<LMPHostType>);
#else

#ifndef LMP_PAIR_DISPERSION_D3_KOKKOS_H
#define LMP_PAIR_DISPERSION_D3_KOKKOS_H

#include "pair_dispersion_d3.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neigh_list_kokkos.h"


static constexpr double K1 = 16.0;
static constexpr double K3 = -4.0;
static constexpr double autoang =  0.52917725 ;
static constexpr double autoev  = 27.21140795 ;

namespace LAMMPS_NS
{

struct params_d3 {
  F_FLOAT cut;
  F_FLOAT cutsq;
  F_FLOAT s6;
  F_FLOAT s8;
  F_FLOAT rs6;
  F_FLOAT rs8;
  F_FLOAT alpha;
  F_FLOAT r0_ab;
  F_FLOAT pad[1];
};

// -----------------------------------------------------------------------------
// Generic bulk-init helper for DualView host mirrors, rank-aware up to 5D
// -----------------------------------------------------------------------------
/*
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
  View1D_F      d_special_lj;
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
    View1D_F d_special_lj_,
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
  View1D_F      d_special_lj;
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
    View1D_F d_special_lj_,
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



struct DC6 {
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

*/
/*
template<typename DualViewType, typename FillerFunc, class DeviceType>
KOKKOS_INLINE_FUNCTION
void init_dualview_from_host(DualViewType &dv, FillerFunc filler)
{
  auto h_view = dv.h_view;

  if constexpr (DualViewType::view_type::Rank == 1) {
    for (size_t i = 0; i < h_view.extent(0); ++i)
      filler(h_view, i);
  }
  else if constexpr (DualViewType::view_type::Rank == 2) {
    for (size_t i = 0; i < h_view.extent(0); ++i)
      for (size_t j = 0; j < h_view.extent(1); ++j)
        filler(h_view, i, j);
  }
  else if constexpr (DualViewType::view_type::Rank == 3) {
    for (size_t i = 0; i < h_view.extent(0); ++i)
      for (size_t j = 0; j < h_view.extent(1); ++j)
        for (size_t k = 0; k < h_view.extent(2); ++k)
          filler(h_view, i, j, k);
  }
  else if constexpr (DualViewType::view_type::Rank == 4) {
    for (size_t i = 0; i < h_view.extent(0); ++i)
      for (size_t j = 0; j < h_view.extent(1); ++j)
        for (size_t k = 0; k < h_view.extent(2); ++k)
          for (size_t l = 0; l < h_view.extent(3); ++l)
            filler(h_view, i, j, k, l);
  }
  else if constexpr (DualViewType::view_type::Rank == 5) {
    for (size_t i = 0; i < h_view.extent(0); ++i)
      for (size_t j = 0; j < h_view.extent(1); ++j)
        for (size_t k = 0; k < h_view.extent(2); ++k)
          for (size_t l = 0; l < h_view.extent(3); ++l)
            for (size_t m = 0; m < h_view.extent(4); ++m)
              filler(h_view, i, j, k, l, m);
  }
  else {
    static_assert(DualViewType::view_type::Rank <= 5, "Rank > 5 not implemented");
  }

  dv.template modify<LMPHostType>();
  dv.template sync<DeviceType>();
}
*/
template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3
{
  
  public:
    // flags for lammps 
    enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
    
    typedef ArrayTypes<DeviceType> AT; 
    // override baseclass methods we are not using 
    PairDispersionD3Kokkos(class LAMMPS *);
    ~PairDispersionD3Kokkos() override;
    //void allocate() override; 
    void allocateKK(); 
    double init_one(int, int) override;
    void init_style() override; 
    void coeff(int narg, char **arg) override;  
    void compute(int eflag, int vflag) override; 
    void settings(int narg, char **arg) override; 
    void calc_coordination_numberKK(); 
    void sync_coeffs_to_device();
    int pack_forward_comm(int, int*, double*, int, int*) override;
    void unpack_forward_comm(int, int, double*) override;
    int pack_reverse_comm(int, int, double*) override;
    void unpack_reverse_comm(int, int*, double*) override;
                              
    int    pack_forward_comm(int n, DAT::tdual_int_1d k_list,
                                    DAT::tdual_xfloat_1d &k_buf,
                                    int /*pbc_flag*/, int /*pbc*/);
    void unpack_forward_comm(int n, int first, 
                                    DAT::tdual_xfloat_1d &k_buf);
    int    pack_reverse_comm(int n, int first, 
                                    DAT::tdual_xfloat_1d &k_buf);
    void unpack_reverse_comm(int n, DAT::tdual_int_1d k_list,
                                    DAT::tdual_xfloat_1d &k_buf);
 
    //double *get_dC6(int, int, double, double) override;
    //double get_dC6(int iat, int jat, double cni, double cnj) override; 
    // Holds all per-typepair constants for D3 zero damping
    /*struct params_d3 
    {
    // Common scalars
      F_FLOAT cut;      // cutoff distance
      F_FLOAT cutsq;    // squared cutoff
      F_FLOAT s6;       // scaling factor for C6 term
      F_FLOAT s8;       // scaling factor for C8 term (if used)
      F_FLOAT rs6;      // damping range scaling for C6
      F_FLOAT rs8;      // damping range scaling for C8
      F_FLOAT alpha;    // damping exponent

      // Typepair‐specific data
      F_FLOAT r0_ab;    // reference R0 for pair (i,j)
//      F_FLOAT c6_ab;    // C6 coefficient for (i,j)
//      F_FLOAT c8_ab;    // C8 coefficient for (i,j) if applicable

      // Padding for alignment if you like (helps coalescing)
      F_FLOAT pad[1];
     }; */
  protected:
  //Kokkos::DualView<F_FLOAT*,         Kokkos::LayoutRight,DeviceType> k_r2r4;   // per-type
  //Kokkos::DualView<F_FLOAT*,         Kokkos::LayoutRight,DeviceType> k_cn; 
  
  typename AT::tdual_xfloat_1d      k_buf;
  typename AT::tdual_xfloat_2d      k_x; 
  typename AT::tdual_ffloat_2d      k_f;
  typename AT::tdual_int_1d         k_list, k_type, k_ilist, k_numneigh, k_special_lj; 
  //typename AT::tdual_double_1d      k_C6Dev, k_cn, k_dc6,  k_r2r4, k_rcov, k_cn_thr, k_rthr;
  typename AT::tdual_double_1d      k_C6Dev, k_cn, k_dc6, k_rcov;
  //typename AT::tdual_int_1d         k_mxci; 
  typename AT::tdual_neighbors_2d   k_firstneigh; 
  
  int communicationStage, neighflag ;
  double k_rthr, k_cn_thr; 
  
  // corresponding views for mxci & c6ab
  // in your header
  Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_params;
  Kokkos::DualView<F_FLOAT*, DeviceType>                         k_r2r4;
  Kokkos::DualView<int*, DeviceType>                             k_mxci;
  Kokkos::DualView<double*****, Kokkos::LayoutRight, DeviceType> k_c6ab;

  void repack_host_to_dual(int ntypes, int gi_max, int gj_max);
};

}


#endif 
#endif
