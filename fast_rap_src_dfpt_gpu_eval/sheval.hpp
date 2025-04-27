#pragma once

#include "common.hpp"
#include "device.hpp"

template <int l_atom_max>
HOST_DEVICE static void SHEval(double sintheta, double costheta, double sinphi, double cosphi, double *pSH);

template <>
HOST_DEVICE void SHEval<0>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  pSH[0] = 0.28209479177387814347;
}

template <>
HOST_DEVICE void SHEval<1>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;
}

template <>
HOST_DEVICE void SHEval<2>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;
}

template <>
HOST_DEVICE void SHEval<3>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fTmpC_m1_1 = -2.2852289973223286808 * fZ2 + 0.45704579946446573616;
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;
  double fTmpB_m2_1 = 1.4453057213202770277 * fZ;
  double fC0_m3 = fX * fC1_m2 - fY * fS1_m2;
  double fS0_m3 = fX * fS1_m2 + fY * fC1_m2;
  double fTmpA_m3_1 = -0.59004358992664351035;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;

  pSH[9] = -fTmpA_m3_1 * fS0_m3;
  pSH[10] = fTmpB_m2_1 * fS1_m2;
  pSH[11] = -fTmpC_m1_1 * fS0_m1;
  pSH[12] = fZ * (1.8658816629505769571 * fZ2 - 1.1195289977703461742);
  pSH[13] = fTmpC_m1_1 * fC0_m1;
  pSH[14] = fTmpB_m2_1 * fC1_m2;
  pSH[15] = fTmpA_m3_1 * fC0_m3;
}

template <>
HOST_DEVICE void SHEval<4>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fTmpC_m1_1 = -2.2852289973223286808 * fZ2 + 0.45704579946446573616;
  double fTmpA_m1_2 = fZ * (-4.6833258049010241757 * fZ2 + 2.0071396306718675039);
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;
  double fTmpB_m2_1 = 1.4453057213202770277 * fZ;
  double fTmpC_m2 = 3.3116114351514600633 * fZ2 - 0.47308734787878000905;
  double fC0_m3 = fX * fC1_m2 - fY * fS1_m2;
  double fS0_m3 = fX * fS1_m2 + fY * fC1_m2;
  double fTmpA_m3_1 = -0.59004358992664351035;
  double fTmpB_m3 = -1.7701307697799305310 * fZ;
  double fC1_m4 = fX * fC0_m3 - fY * fS0_m3;
  double fS1_m4 = fX * fS0_m3 + fY * fC0_m3;
  double fTmpA_m4 = 0.62583573544917613459;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;

  pSH[9] = -fTmpA_m3_1 * fS0_m3;
  pSH[10] = fTmpB_m2_1 * fS1_m2;
  pSH[11] = -fTmpC_m1_1 * fS0_m1;
  pSH[12] = fZ * (1.8658816629505769571 * fZ2 - 1.1195289977703461742);
  pSH[13] = fTmpC_m1_1 * fC0_m1;
  pSH[14] = fTmpB_m2_1 * fC1_m2;
  pSH[15] = fTmpA_m3_1 * fC0_m3;

  pSH[16] = fTmpA_m4 * fS1_m4;
  pSH[17] = -fTmpB_m3 * fS0_m3;
  pSH[18] = fTmpC_m2 * fS1_m2;
  pSH[19] = -fTmpA_m1_2 * fS0_m1;
  pSH[20] = 1.9843134832984429429 * fZ * pSH[12] - 1.0062305898749053634 * pSH[6];
  pSH[21] = fTmpA_m1_2 * fC0_m1;
  pSH[22] = fTmpC_m2 * fC1_m2;
  pSH[23] = fTmpB_m3 * fC0_m3;
  pSH[24] = fTmpA_m4 * fC1_m4;
}

template <>
HOST_DEVICE void SHEval<5>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fTmpC_m1_1 = -2.2852289973223286808 * fZ2 + 0.45704579946446573616;
  double fTmpA_m1_2 = fZ * (-4.6833258049010241757 * fZ2 + 2.0071396306718675039);
  double fTmpB_m1_2 = 2.0310096011589900901 * fZ * fTmpA_m1_2 - 0.99103120896511485334 * fTmpC_m1_1;
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;
  double fTmpB_m2_1 = 1.4453057213202770277 * fZ;
  double fTmpC_m2 = 3.3116114351514600633 * fZ2 - 0.47308734787878000905;
  double fTmpA_m2_2 = fZ * (7.1903051774599856325 * fZ2 - 2.3967683924866618775);
  double fC0_m3 = fX * fC1_m2 - fY * fS1_m2;
  double fS0_m3 = fX * fS1_m2 + fY * fC1_m2;
  double fTmpA_m3_1 = -0.59004358992664351035;
  double fTmpB_m3 = -1.7701307697799305310 * fZ;
  double fTmpC_m3 = -4.4031446949172534892 * fZ2 + 0.48923829943525038768;
  double fC1_m4 = fX * fC0_m3 - fY * fS0_m3;
  double fS1_m4 = fX * fS0_m3 + fY * fC0_m3;
  double fTmpA_m4 = 0.62583573544917613459;
  double fTmpB_m4 = 2.0756623148810412790 * fZ;
  double fC0_m5 = fX * fC1_m4 - fY * fS1_m4;
  double fS0_m5 = fX * fS1_m4 + fY * fC1_m4;
  double fTmpA_m5 = -0.65638205684017010281;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;

  pSH[9] = -fTmpA_m3_1 * fS0_m3;
  pSH[10] = fTmpB_m2_1 * fS1_m2;
  pSH[11] = -fTmpC_m1_1 * fS0_m1;
  pSH[12] = fZ * (1.8658816629505769571 * fZ2 - 1.1195289977703461742);
  pSH[13] = fTmpC_m1_1 * fC0_m1;
  pSH[14] = fTmpB_m2_1 * fC1_m2;
  pSH[15] = fTmpA_m3_1 * fC0_m3;

  pSH[16] = fTmpA_m4 * fS1_m4;
  pSH[17] = -fTmpB_m3 * fS0_m3;
  pSH[18] = fTmpC_m2 * fS1_m2;
  pSH[19] = -fTmpA_m1_2 * fS0_m1;
  pSH[20] = 1.9843134832984429429 * fZ * pSH[12] - 1.0062305898749053634 * pSH[6];
  pSH[21] = fTmpA_m1_2 * fC0_m1;
  pSH[22] = fTmpC_m2 * fC1_m2;
  pSH[23] = fTmpB_m3 * fC0_m3;
  pSH[24] = fTmpA_m4 * fC1_m4;

  pSH[25] = -fTmpA_m5 * fS0_m5;
  pSH[26] = fTmpB_m4 * fS1_m4;
  pSH[27] = -fTmpC_m3 * fS0_m3;
  pSH[28] = fTmpA_m2_2 * fS1_m2;
  pSH[29] = -fTmpB_m1_2 * fS0_m1;
  pSH[30] = 1.9899748742132399095 * fZ * pSH[20] - 1.0028530728448139498 * pSH[12];
  pSH[31] = fTmpB_m1_2 * fC0_m1;
  pSH[32] = fTmpA_m2_2 * fC1_m2;
  pSH[33] = fTmpC_m3 * fC0_m3;
  pSH[34] = fTmpB_m4 * fC1_m4;
  pSH[35] = fTmpA_m5 * fC0_m5;
}

template <>
HOST_DEVICE void SHEval<6>(double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fTmpC_m1_1 = -2.2852289973223286808 * fZ2 + 0.45704579946446573616;
  double fTmpA_m1_2 = fZ * (-4.6833258049010241757 * fZ2 + 2.0071396306718675039);
  double fTmpB_m1_2 = 2.0310096011589900901 * fZ * fTmpA_m1_2 - 0.99103120896511485334 * fTmpC_m1_1;
  double fTmpC_m1_2 = 2.0213149892370277761 * fZ * fTmpB_m1_2 - 0.99522670305623857702 * fTmpA_m1_2;
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;
  double fTmpB_m2_1 = 1.4453057213202770277 * fZ;
  double fTmpC_m2 = 3.3116114351514600633 * fZ2 - 0.47308734787878000905;
  double fTmpA_m2_2 = fZ * (7.1903051774599856325 * fZ2 - 2.3967683924866618775);
  double fTmpB_m2_2 = 2.1139418156609703623 * fZ * fTmpA_m2_2 - 0.97361012046232688422 * fTmpC_m2;
  double fC0_m3 = fX * fC1_m2 - fY * fS1_m2;
  double fS0_m3 = fX * fS1_m2 + fY * fC1_m2;
  double fTmpA_m3_1 = -0.59004358992664351035;
  double fTmpB_m3 = -1.7701307697799305310 * fZ;
  double fTmpC_m3 = -4.4031446949172534892 * fZ2 + 0.48923829943525038768;
  double fTmpA_m3_2 = fZ * (-10.133257854664158491 * fZ2 + 2.7636157785447704974);
  double fC1_m4 = fX * fC0_m3 - fY * fS0_m3;
  double fS1_m4 = fX * fS0_m3 + fY * fC0_m3;
  double fTmpA_m4 = 0.62583573544917613459;
  double fTmpB_m4 = 2.0756623148810412790 * fZ;
  double fTmpC_m4 = 5.5502139080159657518 * fZ2 - 0.50456490072872415925;
  double fC0_m5 = fX * fC1_m4 - fY * fS1_m4;
  double fS0_m5 = fX * fS1_m4 + fY * fC1_m4;
  double fTmpA_m5 = -0.65638205684017010281;
  double fTmpB_m5 = -2.3666191622317520320 * fZ;
  double fC1_m6 = fX * fC0_m5 - fY * fS0_m5;
  double fS1_m6 = fX * fS0_m5 + fY * fC0_m5;
  double fTmpC_m6 = 0.68318410519191432198;

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;

  pSH[9] = -fTmpA_m3_1 * fS0_m3;
  pSH[10] = fTmpB_m2_1 * fS1_m2;
  pSH[11] = -fTmpC_m1_1 * fS0_m1;
  pSH[12] = fZ * (1.8658816629505769571 * fZ2 - 1.1195289977703461742);
  pSH[13] = fTmpC_m1_1 * fC0_m1;
  pSH[14] = fTmpB_m2_1 * fC1_m2;
  pSH[15] = fTmpA_m3_1 * fC0_m3;

  pSH[16] = fTmpA_m4 * fS1_m4;
  pSH[17] = -fTmpB_m3 * fS0_m3;
  pSH[18] = fTmpC_m2 * fS1_m2;
  pSH[19] = -fTmpA_m1_2 * fS0_m1;
  pSH[20] = 1.9843134832984429429 * fZ * pSH[12] - 1.0062305898749053634 * pSH[6];
  pSH[21] = fTmpA_m1_2 * fC0_m1;
  pSH[22] = fTmpC_m2 * fC1_m2;
  pSH[23] = fTmpB_m3 * fC0_m3;
  pSH[24] = fTmpA_m4 * fC1_m4;

  pSH[25] = -fTmpA_m5 * fS0_m5;
  pSH[26] = fTmpB_m4 * fS1_m4;
  pSH[27] = -fTmpC_m3 * fS0_m3;
  pSH[28] = fTmpA_m2_2 * fS1_m2;
  pSH[29] = -fTmpB_m1_2 * fS0_m1;
  pSH[30] = 1.9899748742132399095 * fZ * pSH[20] - 1.0028530728448139498 * pSH[12];
  pSH[31] = fTmpB_m1_2 * fC0_m1;
  pSH[32] = fTmpA_m2_2 * fC1_m2;
  pSH[33] = fTmpC_m3 * fC0_m3;
  pSH[34] = fTmpB_m4 * fC1_m4;
  pSH[35] = fTmpA_m5 * fC0_m5;

  pSH[36] = fTmpC_m6 * fS1_m6;
  pSH[37] = -fTmpB_m5 * fS0_m5;
  pSH[38] = fTmpC_m4 * fS1_m4;
  pSH[39] = -fTmpA_m3_2 * fS0_m3;
  pSH[40] = fTmpB_m2_2 * fS1_m2;
  pSH[41] = -fTmpC_m1_2 * fS0_m1;
  pSH[42] = 1.9930434571835663369 * fZ * pSH[30] - 1.0015420209622192481 * pSH[20];
  pSH[43] = fTmpC_m1_2 * fC0_m1;
  pSH[44] = fTmpB_m2_2 * fC1_m2;
  pSH[45] = fTmpA_m3_2 * fC0_m3;
  pSH[46] = fTmpC_m4 * fC1_m4;
  pSH[47] = fTmpB_m5 * fC0_m5;
  pSH[48] = fTmpC_m6 * fC1_m6;
}

HOST_DEVICE static void
SHEval_s(int lmax, double sintheta, double costheta, double sinphi, double cosphi, double *pSH) {
  switch (lmax) {
  case 0: SHEval<0>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 1: SHEval<1>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 2: SHEval<2>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 3: SHEval<3>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 4: SHEval<4>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 5: SHEval<5>(sintheta, costheta, sinphi, cosphi, pSH); break;
  case 6: SHEval<6>(sintheta, costheta, sinphi, cosphi, pSH); break;
  default: break;
  }
}

#define ACC_ADD(i, val)                                                                                                \
  acc += (spl_param(0, i, i_spl - 1) * ta + spl_param(1, i, i_spl - 1) * tb + spl_param(0, i, i_spl) * tc +            \
          spl_param(1, i, i_spl) * td) *                                                                               \
         val;

template <int l_atom_max>
HOST_DEVICE static double SHEval_spline_vector_v2_n2_ddot_fused(
    int lmax,
    double sintheta,
    double costheta,
    double sinphi,
    double cosphi,
    double r_output,
    int n_vector,
    int n_l_dim,
    int n_grid_dim,
    int n_points,
    const double *spl_param_ptr);

template <>
HOST_DEVICE double SHEval_spline_vector_v2_n2_ddot_fused<-1>(
    int lmax,
    double sintheta,
    double costheta,
    double sinphi,
    double cosphi,
    double r_output,
    int n_vector,
    int n_l_dim,
    int n_grid_dim,
    int n_points,
    const double *spl_param_ptr) {

  constexpr int n_coeff = 2;
  cTMf64<3> TM_INIT(spl_param, n_coeff, n_l_dim, n_grid_dim);

  int i_spl = int(r_output);
  i_spl = max(1, i_spl);
  i_spl = min(n_points - 1, i_spl);
  double t = r_output - i_spl;

  double ta = (t - 1) * (t - 1) * (1 + 2 * t);
  double tb = (t - 1) * (t - 1) * t;
  double tc = t * t * (3 - 2 * t);
  double td = t * t * (t - 1);

  double fX = sintheta * cosphi;
  double fY = sintheta * sinphi;
  double fZ = costheta;
  double fZ2 = fZ * fZ;

  double fC0_m1 = fX;
  double fS0_m1 = fY;
  double fTmpA_m1_1 = -0.48860251190291992159;
  double fTmpB_m1_1 = -1.0925484305920790705 * fZ;
  double fTmpC_m1_1 = -2.2852289973223286808 * fZ2 + 0.45704579946446573616;
  double fTmpA_m1_2 = fZ * (-4.6833258049010241757 * fZ2 + 2.0071396306718675039);
  double fTmpB_m1_2 = 2.0310096011589900901 * fZ * fTmpA_m1_2 - 0.99103120896511485334 * fTmpC_m1_1;
  double fTmpC_m1_2 = 2.0213149892370277761 * fZ * fTmpB_m1_2 - 0.99522670305623857702 * fTmpA_m1_2;
  double fC1_m2 = fX * fC0_m1 - fY * fS0_m1;
  double fS1_m2 = fX * fS0_m1 + fY * fC0_m1;
  double fTmpA_m2_1 = 0.54627421529603953527;
  double fTmpB_m2_1 = 1.4453057213202770277 * fZ;
  double fTmpC_m2 = 3.3116114351514600633 * fZ2 - 0.47308734787878000905;
  double fTmpA_m2_2 = fZ * (7.1903051774599856325 * fZ2 - 2.3967683924866618775);
  double fTmpB_m2_2 = 2.1139418156609703623 * fZ * fTmpA_m2_2 - 0.97361012046232688422 * fTmpC_m2;
  double fC0_m3 = fX * fC1_m2 - fY * fS1_m2;
  double fS0_m3 = fX * fS1_m2 + fY * fC1_m2;
  double fTmpA_m3_1 = -0.59004358992664351035;
  double fTmpB_m3 = -1.7701307697799305310 * fZ;
  double fTmpC_m3 = -4.4031446949172534892 * fZ2 + 0.48923829943525038768;
  double fTmpA_m3_2 = fZ * (-10.133257854664158491 * fZ2 + 2.7636157785447704974);
  double fC1_m4 = fX * fC0_m3 - fY * fS0_m3;
  double fS1_m4 = fX * fS0_m3 + fY * fC0_m3;
  double fTmpA_m4 = 0.62583573544917613459;
  double fTmpB_m4 = 2.0756623148810412790 * fZ;
  double fTmpC_m4 = 5.5502139080159657518 * fZ2 - 0.50456490072872415925;
  double fC0_m5 = fX * fC1_m4 - fY * fS1_m4;
  double fS0_m5 = fX * fS1_m4 + fY * fC1_m4;
  double fTmpA_m5 = -0.65638205684017010281;
  double fTmpB_m5 = -2.3666191622317520320 * fZ;
  double fC1_m6 = fX * fC0_m5 - fY * fS0_m5;
  double fS1_m6 = fX * fS0_m5 + fY * fC0_m5;
  double fTmpC_m6 = 0.68318410519191432198;

  double acc = 0;

  double pSH_6 = 0.94617469575756001809 * fZ2 - 0.31539156525252000603;
  double pSH_12 = fZ * (1.8658816629505769571 * fZ2 - 1.1195289977703461742);
  double pSH_20 = 1.9843134832984429429 * fZ * pSH_12 - 1.0062305898749053634 * pSH_6;
  double pSH_30 = 1.9899748742132399095 * fZ * pSH_20 - 1.0028530728448139498 * pSH_12;
  double pSH_42 = 1.9930434571835663369 * fZ * pSH_30 - 1.0015420209622192481 * pSH_20;

  double pSH[49];

  pSH[0] = 0.28209479177387814347;

  pSH[1] = -fTmpA_m1_1 * fS0_m1;
  pSH[2] = 0.48860251190291992159 * fZ;
  pSH[3] = fTmpA_m1_1 * fC0_m1;

  pSH[4] = fTmpA_m2_1 * fS1_m2;
  pSH[5] = -fTmpB_m1_1 * fS0_m1;
  pSH[6] = pSH_6;
  pSH[7] = fTmpB_m1_1 * fC0_m1;
  pSH[8] = fTmpA_m2_1 * fC1_m2;

  pSH[9] = -fTmpA_m3_1 * fS0_m3;
  pSH[10] = fTmpB_m2_1 * fS1_m2;
  pSH[11] = -fTmpC_m1_1 * fS0_m1;
  pSH[12] = pSH_12;
  pSH[13] = fTmpC_m1_1 * fC0_m1;
  pSH[14] = fTmpB_m2_1 * fC1_m2;
  pSH[15] = fTmpA_m3_1 * fC0_m3;

  pSH[16] = fTmpA_m4 * fS1_m4;
  pSH[17] = -fTmpB_m3 * fS0_m3;
  pSH[18] = fTmpC_m2 * fS1_m2;
  pSH[19] = -fTmpA_m1_2 * fS0_m1;
  pSH[20] = pSH_20;
  pSH[21] = fTmpA_m1_2 * fC0_m1;
  pSH[22] = fTmpC_m2 * fC1_m2;
  pSH[23] = fTmpB_m3 * fC0_m3;
  pSH[24] = fTmpA_m4 * fC1_m4;

  constexpr int len_0 = (0 + 1) * (0 + 1);
  constexpr int len_1 = (1 + 1) * (1 + 1);
  constexpr int len_2 = (2 + 1) * (2 + 1);
  constexpr int len_3 = (3 + 1) * (3 + 1);
  constexpr int len_4 = (4 + 1) * (4 + 1);
  constexpr int len_5 = (5 + 1) * (5 + 1);
  constexpr int len_6 = (6 + 1) * (6 + 1);

  if (lmax >= 5) {
    pSH[25] = -fTmpA_m5 * fS0_m5;
    pSH[26] = fTmpB_m4 * fS1_m4;
    pSH[27] = -fTmpC_m3 * fS0_m3;
    pSH[28] = fTmpA_m2_2 * fS1_m2;
    pSH[29] = -fTmpB_m1_2 * fS0_m1;
    pSH[30] = pSH_30;
    pSH[31] = fTmpB_m1_2 * fC0_m1;
    pSH[32] = fTmpA_m2_2 * fC1_m2;
    pSH[33] = fTmpC_m3 * fC0_m3;
    pSH[34] = fTmpB_m4 * fC1_m4;
    pSH[35] = fTmpA_m5 * fC0_m5;

    pSH[36] = fTmpC_m6 * fS1_m6;
    pSH[37] = -fTmpB_m5 * fS0_m5;
    pSH[38] = fTmpC_m4 * fS1_m4;
    pSH[39] = -fTmpA_m3_2 * fS0_m3;
    pSH[40] = fTmpB_m2_2 * fS1_m2;
    pSH[41] = -fTmpC_m1_2 * fS0_m1;
    pSH[42] = pSH_42;
    pSH[43] = fTmpC_m1_2 * fC0_m1;
    pSH[44] = fTmpB_m2_2 * fC1_m2;
    pSH[45] = fTmpA_m3_2 * fC0_m3;
    pSH[46] = fTmpC_m4 * fC1_m4;
    pSH[47] = fTmpB_m5 * fC0_m5;
    pSH[48] = fTmpC_m6 * fC1_m6;

    XDEF_UNROLL
    for (int i = 0; i < len_5; i++) {
      double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
      double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
      acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
    }
    if (lmax == 5)
      return acc;
    XDEF_UNROLL
    for (int i = len_5; i < len_6; i++) {
      double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
      double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
      acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
    }
    return acc;
  }

  XDEF_UNROLL
  for (int i = 0; i < len_0; i++) {
    double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
    double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
    acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
  }
  if (lmax == 0)
    return acc;
  XDEF_UNROLL
  for (int i = len_0; i < len_1; i++) {
    double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
    double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
    acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
  }
  if (lmax == 1)
    return acc;
  XDEF_UNROLL
  for (int i = len_1; i < len_2; i++) {
    double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
    double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
    acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
  }
  if (lmax == 2)
    return acc;
  XDEF_UNROLL
  for (int i = len_2; i < len_3; i++) {
    double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
    double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
    acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
  }
  if (lmax == 3)
    return acc;
  XDEF_UNROLL
  for (int i = len_3; i < len_4; i++) {
    double2 spl_0 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl - 1))[0];
    double2 spl_1 = reinterpret_cast<const double2 *>(&spl_param(0, i, i_spl))[0];
    acc += (spl_0.x * ta + spl_0.y * tb + spl_1.x * tc + spl_1.y * td) * pSH[i];
  }
  if (lmax == 4)
    return acc;
  return acc;
}
