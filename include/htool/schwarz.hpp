#ifndef SCHWARZ_HPP
#define SCHWARZ_HPP

#include "preconditioner.hpp"

namespace htool{


template<typename T>
class ASM : public Preconditioner<T>{
private:

  int n_local_ovr;
  const std::vector<int> neighbors;
  std::vector<std::vector<int> > intersections;
  std::vector<T> fact;
  std::vector<T> vec_ovr;
  std::vector<std::vector<T>> snd,rcv;
  MPI_Comm comm;



  void synchronize(bool sum){

    fill(vec_ovr.begin()+this->n_local,vec_ovr.end(),0);
    for (int i=0;i<neighbors.size();i++){
      for (int j=0;j<intersections[i].size();j++){
        snd[i][j]=vec_ovr[intersections[i][j]];
      }
    }

    // Communications
		std::vector<MPI_Request> rq(2*neighbors.size());

    for (int i=0;i<neighbors.size();i++){

      MPI_Isend( snd[i].data(), snd[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], 0, comm, &(rq[i]));

		  MPI_Irecv( rcv[i].data(), rcv[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], MPI_ANY_TAG, comm, &(rq[i+neighbors.size()]));
    }
    MPI_Waitall(rq.size(),rq.data(),MPI_STATUSES_IGNORE);

    for (int i=0;i<neighbors.size();i++){
      for (int j=0;j<intersections[i].size();j++){
          vec_ovr[intersections[i][j]]+=rcv[i][j];
      }
    }

  }

public:
  ASM(const IMatrix<T>& mat0, const std::vector<int>&  ovr_subdomain_to_global0, const std::vector<int>& cluster_to_ovr_subdomain0, const std::vector<int>& neighbors0, const std::vector<std::vector<int> >& intersections0, MPI_Comm comm0=MPI_COMM_WORLD):Preconditioner<T>(cluster_to_ovr_subdomain0.size()), n_local_ovr(ovr_subdomain_to_global0.size()), neighbors(neighbors0),fact(ovr_subdomain_to_global0.size()*ovr_subdomain_to_global0.size()), vec_ovr(ovr_subdomain_to_global0.size()), comm(comm0) {

    std::vector<int> renum(n_local_ovr,-1);
    std::vector<int> renum_to_global(n_local_ovr);

    for (int i=0;i<cluster_to_ovr_subdomain0.size();i++){
      renum[cluster_to_ovr_subdomain0[i]]=i;
      renum_to_global[i]=ovr_subdomain_to_global0[cluster_to_ovr_subdomain0[i]];
    }
    int count =cluster_to_ovr_subdomain0.size();
    for (int i=0;i<n_local_ovr;i++){
      if (renum[i]==-1){
        renum[i]=count;
        renum_to_global[count++]=ovr_subdomain_to_global0[i];
      }
    }

    fact = mat0.get_submatrix(renum_to_global,renum_to_global).get_mat();

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

  }
  void num_fact(){
    const char l='L';
    int lda=n_local_ovr;
    int info;
    Lapack<T>::potrf(&l,&(n_local_ovr),fact.data(),&lda,&info);
  }

  void apply(const T* const in, T* const out) {

    // Without overlap to with overlap
    std::copy_n(in,this->n_local,vec_ovr.data());
    // std::fill(vec_ovr.begin(),vec_ovr.end(),0);
    // std::fill_n(vec_ovr.begin(),this->n_local,1);
    synchronize(false);
    // std::cout << vec_ovr<<std::endl;
    //
    const char l='L';
    int lda=n_local_ovr;
    int ldb=n_local_ovr;
    int nrhs =1 ;
    int info;

    Lapack<T>::potrs(&l,&(n_local_ovr),&nrhs,fact.data(),&lda,vec_ovr.data(),&ldb,&info);

    //
    synchronize(true);
    std::copy_n(vec_ovr.data(),this->n_local,out);

  }



};

}
#endif
