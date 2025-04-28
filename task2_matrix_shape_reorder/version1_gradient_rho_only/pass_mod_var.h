#ifndef __PASS_MOD_VAR_H__
#define __PASS_MOD_VAR_H__

// 需要注意，INTEL 编译器定义了 GNUC 这个宏，不能用来判断
// ifort 和 gfortran 都将 Fortran 中模块和变量名小写化了
// 因此后面的模块和变量名得改
#ifdef __INTEL_COMPILER
#define MV(mod_name, var_name) mod_name##_mp_##var_name##_
#else
#define MV(mod_name, var_name) __##mod_name##_MOD_##var_name
#endif

// ------

extern int MV(mpi_tasks, n_tasks);
extern int MV(mpi_tasks, myid);

// ------

#define n_tasks MV(mpi_tasks, n_tasks)
#define myid MV(mpi_tasks, myid)

#endif