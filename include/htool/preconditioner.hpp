#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include "matrix.hpp"
#include "lapack.hpp"
#include "wrapper_mpi.hpp"
#include "wrapper_hpddm.hpp"

namespace htool{


template<template<typename> class LowRankMatrix, typename T>
class DDM{
private:

  int n;
  int n_inside;
  const std::vector<int> neighbors;
  // const std::vector<int> cluster_to_ovr_subdomain;
  std::vector<std::vector<int> > intersections;
  std::vector<T> vec_ovr;
  std::vector<std::vector<T>> snd,rcv;
  HPDDMDense<LowRankMatrix,T> hpddm_op;
  std::vector<T> mat_loc;
  std::vector<double> D;
  MPI_Comm comm;



  // void synchronize(bool sum){
  //
  //   fill(vec_ovr.begin()+n_inside,vec_ovr.end(),0);
  //   for (int i=0;i<neighbors.size();i++){
  //     for (int j=0;j<intersections[i].size();j++){
  //       snd[i][j]=vec_ovr[intersections[i][j]];
  //       // if (!sum && intersections[i][j]>=n_inside)  snd[i][j]=0;
  //     }
  //   }
  //
  //   // Communications
	// 	std::vector<MPI_Request> rq(2*neighbors.size());
  //
  //   for (int i=0;i<neighbors.size();i++){
  //
  //     MPI_Isend( snd[i].data(), snd[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], 0, comm, &(rq[i]));
  //
	// 	  MPI_Irecv( rcv[i].data(), rcv[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], MPI_ANY_TAG, comm, &(rq[i+neighbors.size()]));
  //   }
  //   MPI_Waitall(rq.size(),rq.data(),MPI_STATUSES_IGNORE);
  //
  //   for (int i=0;i<neighbors.size();i++){
  //     for (int j=0;j<intersections[i].size();j++){
  //       // if (sum){
  //         vec_ovr[intersections[i][j]]+=rcv[i][j];
  //       // }
  //       // else {//if (intersections[i][j]>=n_inside){
  //       //   vec_ovr[intersections[i][j]]+=rcv[i][j];
  //       //   // if (rcv[i][j].real()==0) std::cout << "pouet "<< std::endl;
  //       // }
  //     }
  //   }
  //
  // }

public:
  DDM(const IMatrix<T>& mat0, const HMatrix<LowRankMatrix,T>& hmat_0,
    const std::vector<int>&  ovr_subdomain_to_global0,
    const std::vector<int>& cluster_to_ovr_subdomain0,
    const std::vector<int>& neighbors0,
    const std::vector<std::vector<int> >& intersections0,
    MPI_Comm comm0=MPI_COMM_WORLD): hpddm_op(hmat_0), n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n),mat_loc(n*n), D(n), comm(comm0) {

    std::vector<int> renum(n,-1);
    std::vector<int> renum_to_global(n);

    for (int i=0;i<cluster_to_ovr_subdomain0.size();i++){
      renum[cluster_to_ovr_subdomain0[i]]=i;
      renum_to_global[i]=ovr_subdomain_to_global0[cluster_to_ovr_subdomain0[i]];
    }
    int count =cluster_to_ovr_subdomain0.size();
    // std::cout << count << std::endl;
    for (int i=0;i<n;i++){
      if (renum[i]==-1){
        renum[i]=count;
        renum_to_global[count++]=ovr_subdomain_to_global0[i];
      }
    }


// std::cout <<s count <<" "<<n<< std::endl;
    // for (int i=0;i<n;i++){
    //   for (int j=0;j<n;j++){
    //     fact[renum[i]+renum[j]*n]=mat0.get_coef(ovr_subdomain_to_global0[i],ovr_subdomain_to_global0[j]);
    //   }
    // }

    intersections.resize(neighbors.size());
    for (int i=0;i<neighbors.size();i++){
      intersections[i].resize(intersections0[i].size());
      for (int j=0;j<intersections[i].size();j++){
        intersections[i][j]=renum[intersections0[i][j]];
      }
    }

    snd.resize(neighbors.size());
    rcv.resize(neighbors.size());

    for (int i=0;i<neighbors.size();i++){
      snd[i].resize(intersections[i].size());
      rcv[i].resize(intersections[i].size());
    }

    bool sym=false;
    mat_loc= mat0.get_submatrix(renum_to_global,renum_to_global).get_mat();
    hpddm_op.initialize(n, sym, mat_loc.data(), neighbors, intersections);

    fill(D.begin(),D.begin()+n_inside-1,1);
    fill(D.begin()+n_inside,D.end(),0);

    hpddm_op.super::initialize(D.data());
    hpddm_op.callNumfact();
    std::vector<T> x_ref(n,1),f_global(n,1);
    T* const f = &(f_global[0]);
    T* sol = &(x_ref[0]);
    HPDDM::IterativeMethod::solve(hpddm_op, f, sol, 1,comm);

  }
  // void num_fact(){
  //   const char l='L';
  //   int lda=n;
  //   int info;
  //   Lapack<T>::potrf(&l,&n,fact.data(),&lda,&info);
  // }

  // void apply(const T* const in, T* const out) {
  //
  //   // Without overlap to with overlap
  //   std::copy_n(in,n_inside,vec_ovr.data());
  //   // std::fill(vec_ovr.begin(),vec_ovr.end(),0);
  //   // std::fill_n(vec_ovr.begin(),n_inside,1);
  //   synchronize(false);
  //   // std::cout << vec_ovr<<std::endl;
  //   //
  //   const char l='L';
  //   int lda=n;
  //   int ldb=n;
  //   int nrhs =1 ;
  //   int info;
  //
  //   Lapack<T>::potrs(&l,&n,&nrhs,fact.data(),&lda,vec_ovr.data(),&ldb,&info);
  //
  //   //
  //   synchronize(true);
  //   std::copy_n(vec_ovr.data(),n_inside,out);
  //
  // }



};

}
#endif
