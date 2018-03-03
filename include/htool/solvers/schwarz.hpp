#ifndef HTOOL_SCHWARZ_HPP
#define HTOOL_SCHWARZ_HPP

#include "solver.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class Schwarz: public ISolver<LowRankMatrix,T>{
private:
    int n_inside;
    const std::vector<int> neighbors;
    std::vector<std::vector<int> > intersections;
    std::vector<T> vec_ovr;
    std::vector<T> mat_loc;
    std::vector<double> D;

public:
    // Without overlap
    Schwarz(const HMatrix<LowRankMatrix,T>& hmat_0):ISolver<LowRankMatrix,T>(hmat_0), n_inside(hmat_0.get_local_size()),mat_loc(this->n*this->n),D(this->n){
        int n= this->n;

        // Timing
        std::vector<double> mytime(2), maxtime(2), meantime(2);
        double time = MPI_Wtime();

        // Building Ai
        bool sym=false;
        const std::vector<LowRankMatrix<T>*>& MyDiagFarFieldMats = this->hpddm_op.HA.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyDiagNearFieldMats= this->hpddm_op.HA.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int i=0;i<MyDiagNearFieldMats.size();i++){
            const SubMatrix<T>& submat = *(MyDiagNearFieldMats[i]);
            int local_nr = submat.nb_rows();
            int local_nc = submat.nb_cols();
            int offset_i = submat.get_offset_i()-this->hpddm_op.HA.get_local_offset();;
            int offset_j = submat.get_offset_j()-this->hpddm_op.HA.get_local_offset();
            for (int i=0;i<local_nc;i++){
                std::copy_n(&(submat(0,i)),local_nr,&mat_loc[offset_i+(offset_j+i)*n]);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n,n);
        for (int i=0;i<MyDiagFarFieldMats.size();i++){
            const LowRankMatrix<T>& lmat = *(MyDiagFarFieldMats[i]);
            int local_nr = lmat.nb_rows();
            int local_nc = lmat.nb_cols();
            int offset_i = lmat.get_offset_i()-this->hpddm_op.HA.get_local_offset();
            int offset_j = lmat.get_offset_j()-this->hpddm_op.HA.get_local_offset();;
            FarFielBlock.resize(local_nr,local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0,0)));
            for (int i=0;i<local_nc;i++){
                std::copy_n(&(FarFielBlock(0,i)),local_nr,&mat_loc[offset_i+(offset_j+i)*n]);
            }
        }

        std::vector<int> neighbors;
        std::vector<std::vector<int> > intersections;
        this->hpddm_op.initialize(n, sym, mat_loc.data(), neighbors, intersections);

        fill(D.begin(),D.begin()+n_inside,1);
        fill(D.begin()+n_inside,D.end(),0);

        this->hpddm_op.HPDDMDense<LowRankMatrix,T>::super::super::initialize(D.data());
        mytime[0] =  MPI_Wtime() - time;
        time = MPI_Wtime();
        this->hpddm_op.callNumfact();
        mytime[1] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0,this->comm);
        MPI_Reduce(&(mytime[0]), &(meantime[0]), 2, MPI_DOUBLE, MPI_SUM, 0,this->comm);
        meantime /= hmat_0.get_sizeworld();

        this->infos["DDM_setup_mean"]= NbrToStr(meantime[0]);
        this->infos["DDM_setup_max" ]= NbrToStr(maxtime[0]);
        this->infos["DDM_facto_mean"]= NbrToStr(meantime[1]);
        this->infos["DDM_facto_max" ]= NbrToStr(maxtime[1]);
    }

    Schwarz(const IMatrix<T>& mat0, const HMatrix<LowRankMatrix,T>& hmat_0, const std::vector<int>&  ovr_subdomain_to_global0, const std::vector<int>& cluster_to_ovr_subdomain0, const std::vector<int>& neighbors0, const std::vector<std::vector<int> >& intersections0):ISolver<LowRankMatrix,T>(hmat_0), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(this->n),mat_loc(this->n*this->n), D(this->n) {

        int n = this->n;

        // Timing
        std::vector<double> mytime(2), maxtime(2), meantime(2);
        double time = MPI_Wtime();

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


        intersections.resize(neighbors.size());
        for (int i=0;i<neighbors.size();i++){
            intersections[i].resize(intersections0[i].size());
            for (int j=0;j<intersections[i].size();j++){
                intersections[i][j]=renum[intersections0[i][j]];
            }
        }

        // Building Ai
        bool sym=false;
        const std::vector<LowRankMatrix<T>*>& MyDiagFarFieldMats = this->hpddm_op.HA.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyDiagNearFieldMats= this->hpddm_op.HA.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int i=0;i<MyDiagNearFieldMats.size();i++){
            const SubMatrix<T>& submat = *(MyDiagNearFieldMats[i]);
            int local_nr = submat.nb_rows();
            int local_nc = submat.nb_cols();
            int offset_i = submat.get_offset_i()-this->hpddm_op.HA.get_local_offset();;
            int offset_j = submat.get_offset_j()-this->hpddm_op.HA.get_local_offset();
            for (int i=0;i<local_nc;i++){
                std::copy_n(&(submat(0,i)),local_nr,&mat_loc[offset_i+(offset_j+i)*n]);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n,n);
        for (int i=0;i<MyDiagFarFieldMats.size();i++){
            const LowRankMatrix<T>& lmat = *(MyDiagFarFieldMats[i]);
            int local_nr = lmat.nb_rows();
            int local_nc = lmat.nb_cols();
            int offset_i = lmat.get_offset_i()-this->hpddm_op.HA.get_local_offset();
            int offset_j = lmat.get_offset_j()-this->hpddm_op.HA.get_local_offset();;
            FarFielBlock.resize(local_nr,local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0,0)));
            for (int i=0;i<local_nc;i++){
                std::copy_n(&(FarFielBlock(0,i)),local_nr,&mat_loc[offset_i+(offset_j+i)*n]);
            }
        }


        // Overlap
        std::vector<T> horizontal_block(n-n_inside,n_inside),vertical_block(n,n-n_inside);
        horizontal_block = mat0.get_submatrix(std::vector<int>(renum_to_global.begin()+n_inside,renum_to_global.end()),std::vector<int>(renum_to_global.begin(),renum_to_global.begin()+n_inside)).get_mat();
        vertical_block = mat0.get_submatrix(renum_to_global,std::vector<int>(renum_to_global.begin()+n_inside,renum_to_global.end())).get_mat();
        for (int j=0;j<n_inside;j++){
            std::copy_n(horizontal_block.begin()+j*(n-n_inside),n-n_inside,&mat_loc[n_inside+j*n]);
        }
        for (int j=n_inside;j<n;j++){
            std::copy_n(vertical_block.begin()+(j-n_inside)*n,n,&mat_loc[j*n]);
        }

        // mat_loc= mat0.get_submatrix(renum_to_global,renum_to_global).get_mat();

        this->hpddm_op.initialize(n, sym, mat_loc.data(), neighbors, intersections);

        fill(D.begin(),D.begin()+n_inside,1);
        fill(D.begin()+n_inside,D.end(),0);

        this->hpddm_op.HPDDMDense<LowRankMatrix,T>::super::super::initialize(D.data());
        mytime[0] =  MPI_Wtime() - time;
        time = MPI_Wtime();
        this->hpddm_op.callNumfact();
        mytime[1] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0,this->comm);
        MPI_Reduce(&(mytime[0]), &(meantime[0]), 2, MPI_DOUBLE, MPI_SUM, 0,this->comm);
        meantime /= hmat_0.get_sizeworld();

        this->infos["DDM_setup_mean"]= NbrToStr(meantime[0]);
        this->infos["DDM_setup_max" ]= NbrToStr(maxtime[0]);
        this->infos["DDM_facto_mean"]= NbrToStr(meantime[1]);
        this->infos["DDM_facto_max" ]= NbrToStr(maxtime[1]);

    }

    void solve(const T* const rhs, T* const x, const int& mu=1 ){
        //
        int n= this->n;
        int rankWorld = this->hpddm_op.HA.get_rankworld();
        int sizeWorld = this->hpddm_op.HA.get_sizeworld();
        int offset  = this->hpddm_op.HA.get_local_offset();
        int size    = this->hpddm_op.HA.get_local_size();
        int nb_cols = this->hpddm_op.HA.nb_cols();
        int nb_rows = this->hpddm_op.HA.nb_rows();
        double time = MPI_Wtime();

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n*mu,0);
        std::vector<T> local_rhs(n*mu,0);
        this->hpddm_op.in_global->resize(nb_cols*(mu==1 ? 1 : 2*mu));
        this->hpddm_op.buffer->resize(n_inside*(mu==1 ? 1 : 2*mu));

        // TODO blocking ?
        for (int i=0;i<mu;i++){
            // Permutation
            this->hpddm_op.HA.source_to_cluster_permutation(rhs+i*nb_cols,rhs_perm.data());

            std::copy_n(rhs_perm.begin()+offset,n_inside,local_rhs.begin()+i*n);
        }

        // TODO avoid com here
        // for (int i=0;i<n-n_inside;i++){
        //   local_rhs[i]=rhs_perm[]
        // }
        this->hpddm_op.exchange(local_rhs.data(), mu);

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(this->hpddm_op, local_rhs.data(), x_local.data(), mu,this->comm);

        // Delete the overlap (useful only when mu>1 and n!=n_inside)
        for (int i=0;i<mu;i++){
            std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        }

        // Local to global
        // this->hpddm_op.HA.local_to_global(x_local.data(),this->hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int>  displs(sizeWorld);

        displs[0] = 0;

        for (int i=0; i<sizeWorld; i++) {
            recvcounts[i] = (this->hpddm_op.HA.get_MasterOffset_t(i).second)*mu;
            if (i > 0)
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        MPI_Allgatherv(local_rhs.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), this->hpddm_op.in_global->data() + (mu==1 ? 0 : mu*nb_rows), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), this->comm);

        //

        for (int i=0 ;i<mu;i++){
            if (mu!=1){
                for (int j=0; j<sizeWorld;j++){
                    std::copy_n(this->hpddm_op.in_global->data()+mu*nb_rows+displs[j]+i*recvcounts[j]/mu,recvcounts[j]/mu,this->hpddm_op.in_global->data()+i*nb_rows+displs[j]/mu);
                }
            }

            // Permutation
            this->hpddm_op.HA.cluster_to_target_permutation(this->hpddm_op.in_global->data()+i*nb_rows,x+i*nb_rows);
        }


        // Timing
        HPDDM::Option& opt = *HPDDM::Option::get();
        time = MPI_Wtime()-time;
        this->infos["Solve"] = NbrToStr(time);
        this->infos["Nb_it"] = NbrToStr(nb_it);
        this->infos["mean_time_mat_vec_prod"] = NbrToStr(StrToNbr<double>(this->hpddm_op.HA.get_infos("total_time_mat_vec_prod"))/StrToNbr<double>(this->hpddm_op.HA.get_infos("nbr_mat_vec_prod")));
        switch (opt.val("schwarz_method",0)) {
            case HPDDM_SCHWARZ_METHOD_NONE:
            this->infos["Precond"] = "none";
            break;
            case HPDDM_SCHWARZ_METHOD_RAS:
            this->infos["Precond"] = "ras";
            break;
            case HPDDM_SCHWARZ_METHOD_ASM:
            this->infos["Precond"] = "asm";
            break;
            case HPDDM_SCHWARZ_METHOD_OSM:
            this->infos["Precond"] = "osm";
            break;
            case HPDDM_SCHWARZ_METHOD_ORAS:
            this->infos["Precond"] = "asm";
            break;
            case HPDDM_SCHWARZ_METHOD_SORAS:
            this->infos["Precond"] = "osm";
            break;
        }
    }

};

} //namespace
#endif
