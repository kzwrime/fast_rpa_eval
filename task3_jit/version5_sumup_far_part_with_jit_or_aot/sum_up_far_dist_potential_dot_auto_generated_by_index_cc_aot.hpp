#include "device.hpp"

#define p0(x) 1
#define p1(x) (x)
#define p2(x) (x * x)
#define p3(x) (x * x * x)
#define p4(x) (x * x * x * x)
#define p5(x) (x * x * x * x * x)
#define p6(x) (x * x * x * x * x * x)
#define p7(x) (x * x * x * x * x * x * x)

// template <int l_max>
__device__ double sum_up_far_dist_potential_dot_spec(const double dir[3], const double *Fp, const double *multipole_c, int l_max) {
  double x = dir[0];
  double y = dir[1];
  double z = dir[2];

  double acc = 0;
  acc += p0(x) * p0(y) * p0(z) * Fp[0] * multipole_c[0];
  if (l_max == 0)
    return acc;
  acc += p0(x) * p1(y) * p0(z) * Fp[1] * multipole_c[1];
  acc += p0(x) * p0(y) * p1(z) * Fp[1] * multipole_c[2];
  acc += p1(x) * p0(y) * p0(z) * Fp[1] * multipole_c[3];
  if (l_max == 1)
    return acc;
  acc += p1(x) * p1(y) * p0(z) * Fp[2] * multipole_c[4];
  acc += p0(x) * p1(y) * p1(z) * Fp[2] * multipole_c[5];
  acc += p0(x) * p0(y) * p0(z) * Fp[1] * multipole_c[6];
  acc += p0(x) * p0(y) * p2(z) * Fp[2] * multipole_c[7];
  acc += p1(x) * p0(y) * p1(z) * Fp[2] * multipole_c[8];
  acc += p0(x) * p0(y) * p0(z) * Fp[1] * multipole_c[9];
  acc += p0(x) * p0(y) * p2(z) * Fp[2] * multipole_c[10];
  acc += p0(x) * p2(y) * p0(z) * Fp[2] * multipole_c[11];
  if (l_max == 2)
    return acc;
  acc += p0(x) * p1(y) * p0(z) * Fp[2] * multipole_c[12];
  acc += p0(x) * p1(y) * p2(z) * Fp[3] * multipole_c[13];
  acc += p0(x) * p3(y) * p0(z) * Fp[3] * multipole_c[14];
  acc += p1(x) * p1(y) * p1(z) * Fp[3] * multipole_c[15];
  acc += p0(x) * p1(y) * p0(z) * Fp[2] * multipole_c[16];
  acc += p0(x) * p1(y) * p2(z) * Fp[3] * multipole_c[17];
  acc += p0(x) * p0(y) * p1(z) * Fp[2] * multipole_c[18];
  acc += p0(x) * p0(y) * p3(z) * Fp[3] * multipole_c[19];
  acc += p1(x) * p0(y) * p0(z) * Fp[2] * multipole_c[20];
  acc += p1(x) * p0(y) * p2(z) * Fp[3] * multipole_c[21];
  acc += p0(x) * p0(y) * p1(z) * Fp[2] * multipole_c[22];
  acc += p0(x) * p0(y) * p3(z) * Fp[3] * multipole_c[23];
  acc += p0(x) * p2(y) * p1(z) * Fp[3] * multipole_c[24];
  acc += p1(x) * p0(y) * p0(z) * Fp[2] * multipole_c[25];
  acc += p1(x) * p0(y) * p2(z) * Fp[3] * multipole_c[26];
  acc += p1(x) * p2(y) * p0(z) * Fp[3] * multipole_c[27];
  if (l_max == 3)
    return acc;
  acc += p1(x) * p1(y) * p0(z) * Fp[3] * multipole_c[28];
  acc += p1(x) * p1(y) * p2(z) * Fp[4] * multipole_c[29];
  acc += p1(x) * p3(y) * p0(z) * Fp[4] * multipole_c[30];
  acc += p0(x) * p1(y) * p1(z) * Fp[3] * multipole_c[31];
  acc += p0(x) * p1(y) * p3(z) * Fp[4] * multipole_c[32];
  acc += p0(x) * p3(y) * p1(z) * Fp[4] * multipole_c[33];
  acc += p1(x) * p1(y) * p0(z) * Fp[3] * multipole_c[34];
  acc += p1(x) * p1(y) * p2(z) * Fp[4] * multipole_c[35];
  acc += p0(x) * p1(y) * p1(z) * Fp[3] * multipole_c[36];
  acc += p0(x) * p1(y) * p3(z) * Fp[4] * multipole_c[37];
  acc += p0(x) * p0(y) * p0(z) * Fp[2] * multipole_c[38];
  acc += p0(x) * p0(y) * p2(z) * Fp[3] * multipole_c[39];
  acc += p0(x) * p0(y) * p4(z) * Fp[4] * multipole_c[40];
  acc += p1(x) * p0(y) * p1(z) * Fp[3] * multipole_c[41];
  acc += p1(x) * p0(y) * p3(z) * Fp[4] * multipole_c[42];
  acc += p0(x) * p0(y) * p0(z) * Fp[2] * multipole_c[43];
  acc += p0(x) * p0(y) * p2(z) * Fp[3] * multipole_c[44];
  acc += p0(x) * p0(y) * p4(z) * Fp[4] * multipole_c[45];
  acc += p0(x) * p2(y) * p0(z) * Fp[3] * multipole_c[46];
  acc += p0(x) * p2(y) * p2(z) * Fp[4] * multipole_c[47];
  acc += p1(x) * p0(y) * p1(z) * Fp[3] * multipole_c[48];
  acc += p1(x) * p0(y) * p3(z) * Fp[4] * multipole_c[49];
  acc += p1(x) * p2(y) * p1(z) * Fp[4] * multipole_c[50];
  acc += p0(x) * p0(y) * p0(z) * Fp[2] * multipole_c[51];
  acc += p0(x) * p0(y) * p2(z) * Fp[3] * multipole_c[52];
  acc += p0(x) * p0(y) * p4(z) * Fp[4] * multipole_c[53];
  acc += p0(x) * p2(y) * p0(z) * Fp[3] * multipole_c[54];
  acc += p0(x) * p2(y) * p2(z) * Fp[4] * multipole_c[55];
  acc += p0(x) * p4(y) * p0(z) * Fp[4] * multipole_c[56];
  if (l_max == 4)
    return acc;
  acc += p0(x) * p1(y) * p0(z) * Fp[3] * multipole_c[57];
  acc += p0(x) * p1(y) * p2(z) * Fp[4] * multipole_c[58];
  acc += p0(x) * p1(y) * p4(z) * Fp[5] * multipole_c[59];
  acc += p0(x) * p3(y) * p0(z) * Fp[4] * multipole_c[60];
  acc += p0(x) * p3(y) * p2(z) * Fp[5] * multipole_c[61];
  acc += p0(x) * p5(y) * p0(z) * Fp[5] * multipole_c[62];
  acc += p1(x) * p1(y) * p1(z) * Fp[4] * multipole_c[63];
  acc += p1(x) * p1(y) * p3(z) * Fp[5] * multipole_c[64];
  acc += p1(x) * p3(y) * p1(z) * Fp[5] * multipole_c[65];
  acc += p0(x) * p1(y) * p0(z) * Fp[3] * multipole_c[66];
  acc += p0(x) * p1(y) * p2(z) * Fp[4] * multipole_c[67];
  acc += p0(x) * p1(y) * p4(z) * Fp[5] * multipole_c[68];
  acc += p0(x) * p3(y) * p0(z) * Fp[4] * multipole_c[69];
  acc += p0(x) * p3(y) * p2(z) * Fp[5] * multipole_c[70];
  acc += p1(x) * p1(y) * p1(z) * Fp[4] * multipole_c[71];
  acc += p1(x) * p1(y) * p3(z) * Fp[5] * multipole_c[72];
  acc += p0(x) * p1(y) * p0(z) * Fp[3] * multipole_c[73];
  acc += p0(x) * p1(y) * p2(z) * Fp[4] * multipole_c[74];
  acc += p0(x) * p1(y) * p4(z) * Fp[5] * multipole_c[75];
  acc += p0(x) * p0(y) * p1(z) * Fp[3] * multipole_c[76];
  acc += p0(x) * p0(y) * p3(z) * Fp[4] * multipole_c[77];
  acc += p0(x) * p0(y) * p5(z) * Fp[5] * multipole_c[78];
  acc += p1(x) * p0(y) * p0(z) * Fp[3] * multipole_c[79];
  acc += p1(x) * p0(y) * p2(z) * Fp[4] * multipole_c[80];
  acc += p1(x) * p0(y) * p4(z) * Fp[5] * multipole_c[81];
  acc += p0(x) * p0(y) * p1(z) * Fp[3] * multipole_c[82];
  acc += p0(x) * p0(y) * p3(z) * Fp[4] * multipole_c[83];
  acc += p0(x) * p0(y) * p5(z) * Fp[5] * multipole_c[84];
  acc += p0(x) * p2(y) * p1(z) * Fp[4] * multipole_c[85];
  acc += p0(x) * p2(y) * p3(z) * Fp[5] * multipole_c[86];
  acc += p1(x) * p0(y) * p0(z) * Fp[3] * multipole_c[87];
  acc += p1(x) * p0(y) * p2(z) * Fp[4] * multipole_c[88];
  acc += p1(x) * p0(y) * p4(z) * Fp[5] * multipole_c[89];
  acc += p1(x) * p2(y) * p0(z) * Fp[4] * multipole_c[90];
  acc += p1(x) * p2(y) * p2(z) * Fp[5] * multipole_c[91];
  acc += p0(x) * p0(y) * p1(z) * Fp[3] * multipole_c[92];
  acc += p0(x) * p0(y) * p3(z) * Fp[4] * multipole_c[93];
  acc += p0(x) * p0(y) * p5(z) * Fp[5] * multipole_c[94];
  acc += p0(x) * p2(y) * p1(z) * Fp[4] * multipole_c[95];
  acc += p0(x) * p2(y) * p3(z) * Fp[5] * multipole_c[96];
  acc += p0(x) * p4(y) * p1(z) * Fp[5] * multipole_c[97];
  acc += p1(x) * p0(y) * p0(z) * Fp[3] * multipole_c[98];
  acc += p1(x) * p0(y) * p2(z) * Fp[4] * multipole_c[99];
  acc += p1(x) * p0(y) * p4(z) * Fp[5] * multipole_c[100];
  acc += p1(x) * p2(y) * p0(z) * Fp[4] * multipole_c[101];
  acc += p1(x) * p2(y) * p2(z) * Fp[5] * multipole_c[102];
  acc += p1(x) * p4(y) * p0(z) * Fp[5] * multipole_c[103];
  if (l_max == 5)
    return acc;
  acc += p1(x) * p1(y) * p0(z) * Fp[4] * multipole_c[104];
  acc += p1(x) * p1(y) * p2(z) * Fp[5] * multipole_c[105];
  acc += p1(x) * p1(y) * p4(z) * Fp[6] * multipole_c[106];
  acc += p1(x) * p3(y) * p0(z) * Fp[5] * multipole_c[107];
  acc += p1(x) * p3(y) * p2(z) * Fp[6] * multipole_c[108];
  acc += p1(x) * p5(y) * p0(z) * Fp[6] * multipole_c[109];
  acc += p0(x) * p1(y) * p1(z) * Fp[4] * multipole_c[110];
  acc += p0(x) * p1(y) * p3(z) * Fp[5] * multipole_c[111];
  acc += p0(x) * p1(y) * p5(z) * Fp[6] * multipole_c[112];
  acc += p0(x) * p3(y) * p1(z) * Fp[5] * multipole_c[113];
  acc += p0(x) * p3(y) * p3(z) * Fp[6] * multipole_c[114];
  acc += p0(x) * p5(y) * p1(z) * Fp[6] * multipole_c[115];
  acc += p1(x) * p1(y) * p0(z) * Fp[4] * multipole_c[116];
  acc += p1(x) * p1(y) * p2(z) * Fp[5] * multipole_c[117];
  acc += p1(x) * p1(y) * p4(z) * Fp[6] * multipole_c[118];
  acc += p1(x) * p3(y) * p0(z) * Fp[5] * multipole_c[119];
  acc += p1(x) * p3(y) * p2(z) * Fp[6] * multipole_c[120];
  acc += p0(x) * p1(y) * p1(z) * Fp[4] * multipole_c[121];
  acc += p0(x) * p1(y) * p3(z) * Fp[5] * multipole_c[122];
  acc += p0(x) * p1(y) * p5(z) * Fp[6] * multipole_c[123];
  acc += p0(x) * p3(y) * p1(z) * Fp[5] * multipole_c[124];
  acc += p0(x) * p3(y) * p3(z) * Fp[6] * multipole_c[125];
  acc += p1(x) * p1(y) * p0(z) * Fp[4] * multipole_c[126];
  acc += p1(x) * p1(y) * p2(z) * Fp[5] * multipole_c[127];
  acc += p1(x) * p1(y) * p4(z) * Fp[6] * multipole_c[128];
  acc += p0(x) * p1(y) * p1(z) * Fp[4] * multipole_c[129];
  acc += p0(x) * p1(y) * p3(z) * Fp[5] * multipole_c[130];
  acc += p0(x) * p1(y) * p5(z) * Fp[6] * multipole_c[131];
  acc += p0(x) * p0(y) * p0(z) * Fp[3] * multipole_c[132];
  acc += p0(x) * p0(y) * p2(z) * Fp[4] * multipole_c[133];
  acc += p0(x) * p0(y) * p4(z) * Fp[5] * multipole_c[134];
  acc += p0(x) * p0(y) * p6(z) * Fp[6] * multipole_c[135];
  acc += p1(x) * p0(y) * p1(z) * Fp[4] * multipole_c[136];
  acc += p1(x) * p0(y) * p3(z) * Fp[5] * multipole_c[137];
  acc += p1(x) * p0(y) * p5(z) * Fp[6] * multipole_c[138];
  acc += p0(x) * p0(y) * p0(z) * Fp[3] * multipole_c[139];
  acc += p0(x) * p0(y) * p2(z) * Fp[4] * multipole_c[140];
  acc += p0(x) * p0(y) * p4(z) * Fp[5] * multipole_c[141];
  acc += p0(x) * p0(y) * p6(z) * Fp[6] * multipole_c[142];
  acc += p0(x) * p2(y) * p0(z) * Fp[4] * multipole_c[143];
  acc += p0(x) * p2(y) * p2(z) * Fp[5] * multipole_c[144];
  acc += p0(x) * p2(y) * p4(z) * Fp[6] * multipole_c[145];
  acc += p1(x) * p0(y) * p1(z) * Fp[4] * multipole_c[146];
  acc += p1(x) * p0(y) * p3(z) * Fp[5] * multipole_c[147];
  acc += p1(x) * p0(y) * p5(z) * Fp[6] * multipole_c[148];
  acc += p1(x) * p2(y) * p1(z) * Fp[5] * multipole_c[149];
  acc += p1(x) * p2(y) * p3(z) * Fp[6] * multipole_c[150];
  acc += p0(x) * p0(y) * p0(z) * Fp[3] * multipole_c[151];
  acc += p0(x) * p0(y) * p2(z) * Fp[4] * multipole_c[152];
  acc += p0(x) * p0(y) * p4(z) * Fp[5] * multipole_c[153];
  acc += p0(x) * p0(y) * p6(z) * Fp[6] * multipole_c[154];
  acc += p0(x) * p2(y) * p0(z) * Fp[4] * multipole_c[155];
  acc += p0(x) * p2(y) * p2(z) * Fp[5] * multipole_c[156];
  acc += p0(x) * p2(y) * p4(z) * Fp[6] * multipole_c[157];
  acc += p0(x) * p4(y) * p0(z) * Fp[5] * multipole_c[158];
  acc += p0(x) * p4(y) * p2(z) * Fp[6] * multipole_c[159];
  acc += p1(x) * p0(y) * p1(z) * Fp[4] * multipole_c[160];
  acc += p1(x) * p0(y) * p3(z) * Fp[5] * multipole_c[161];
  acc += p1(x) * p0(y) * p5(z) * Fp[6] * multipole_c[162];
  acc += p1(x) * p2(y) * p1(z) * Fp[5] * multipole_c[163];
  acc += p1(x) * p2(y) * p3(z) * Fp[6] * multipole_c[164];
  acc += p1(x) * p4(y) * p1(z) * Fp[6] * multipole_c[165];
  acc += p0(x) * p0(y) * p0(z) * Fp[3] * multipole_c[166];
  acc += p0(x) * p0(y) * p2(z) * Fp[4] * multipole_c[167];
  acc += p0(x) * p0(y) * p4(z) * Fp[5] * multipole_c[168];
  acc += p0(x) * p0(y) * p6(z) * Fp[6] * multipole_c[169];
  acc += p0(x) * p2(y) * p0(z) * Fp[4] * multipole_c[170];
  acc += p0(x) * p2(y) * p2(z) * Fp[5] * multipole_c[171];
  acc += p0(x) * p2(y) * p4(z) * Fp[6] * multipole_c[172];
  acc += p0(x) * p4(y) * p0(z) * Fp[5] * multipole_c[173];
  acc += p0(x) * p4(y) * p2(z) * Fp[6] * multipole_c[174];
  acc += p0(x) * p6(y) * p0(z) * Fp[6] * multipole_c[175];
  if (l_max == 6)
    return acc;
  return acc;
}

// __device__ double sum_up_far_dist_potential_dot(
//     int l_max,
//     int dir[3],
//     const double* Fp,
//     const double* multipole_c
// ){
//     if(l_max == 0){}
//     else if(l_max == 1){}
// }