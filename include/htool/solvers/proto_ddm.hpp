#ifndef HTOOL_PROTO_DDM_HPP
#define HTOOL_PROTO_DDM_HPP

#include "../types/matrix.hpp"
#include "../types/hmatrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../wrappers/wrapper_hpddm.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class Proto_DDM{
private:
    int n;
    int n_inside;
    const std::vector<int> neighbors;
    std::vector<std::vector<int> > intersections;
    std::vector<T> vec_ovr;
    std::vector<T> mat_loc;
    std::vector<double> D;
    const MPI_Comm& comm;
    mutable std::map<std::string, std::string> infos;
    std::vector<std::vector<T>> snd,rcv;
    std::vector<int> _ipiv;
    std::vector<int> _ipiv_coarse;
    int nevi;
    std::vector<T> evi;
    std::vector<T> E;
    Matrix<T>& evp;
    const HMatrix<LowRankMatrix,T>& hmat;
    double timing_one_level;
    double timing_Q;

    void synchronize(bool scaled){

        // Partition de l'unité
        if (scaled)     fill(vec_ovr.begin()+n_inside,vec_ovr.end(),0);
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

    Proto_DDM(const IMatrix<T>& mat0, const HMatrix<LowRankMatrix,T>& hmat_0,
    const std::vector<int>&  ovr_subdomain_to_global0,
    const std::vector<int>& cluster_to_ovr_subdomain0,
    const std::vector<int>& neighbors0,
    const std::vector<std::vector<int> >& intersections0, Matrix<T>& evp0):  n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n),mat_loc(n*n), D(n), comm(hmat_0.get_comm()), _ipiv(n), evp(evp0), hmat(hmat_0) {

        // Timing
        std::vector<double> mytime(5), maxtime(5), meantime(5);
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
        const std::vector<LowRankMatrix<T>*>& MyDiagFarFieldMats = hmat_0.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyDiagNearFieldMats= hmat_0.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int i=0;i<MyDiagNearFieldMats.size();i++){
          const SubMatrix<T>& submat = *(MyDiagNearFieldMats[i]);
          int local_nr = submat.nb_rows();
          int local_nc = submat.nb_cols();
          int offset_i = submat.get_offset_i()-hmat_0.get_local_offset();
          int offset_j = submat.get_offset_j()-hmat_0.get_local_offset();
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
          int offset_i = lmat.get_offset_i()-hmat_0.get_local_offset();
          int offset_j = lmat.get_offset_j()-hmat_0.get_local_offset();;
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


        mytime[0] =  MPI_Wtime() - time;

        // for (int i=0;i<n;i++){
        //     for (int j=0;j<n;j++){
        //         mat_loc[i+j*n]=(i==j);
        //     }
        // }
        // LU
        // const char l='L';
        int lda=n;
        int info;

        HPDDM::Lapack<T>::getrf(&n,&n,mat_loc.data(),&lda,_ipiv.data(),&info);

        mytime[1] = MPI_Wtime() - time;
        time = MPI_Wtime();
        // delete [] _ipiv;
// std::cout << "info : " << info <<std::endl;

        //

        snd.resize(neighbors.size());
        rcv.resize(neighbors.size());

        for (int i=0;i<neighbors.size();i++){
          snd[i].resize(intersections[i].size());
          rcv[i].resize(intersections[i].size());
        }

        // Local eigenvalue problem
        int n_global= hmat_0.nb_cols();
        int sizeWorld = hmat_0.get_sizeworld();
        int rankWorld = hmat_0.get_rankworld();
        int ldvl = n, ldvr = n, lwork=-1;
        std::vector<T> work(n);
        std::vector<double> rwork(2*n);
        std::vector<T> w(n);
        std::vector<T> vl(n*n), vr(n*n);
        HPDDM::Lapack<T>::geev( "N", "Vectors", &n, evp.data(), &lda, w.data(),nullptr , vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::geev( "N", "Vectors", &n, evp.data(), &lda, w.data(),nullptr , vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );


        mytime[2] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // std::cout << w[2]<<std::endl;
        // if (hmat_0.get_rankworld())
        // if (rankWorld==1){
        //     int select = 2;
        //     for (int i=0;i<n;i++){
        //         std::cout << vr[i+n*select] << " ";
        //     }
        // }
        // std::cout << std::endl;
        std::vector<int> perm1(hmat_0.nb_cols());
        for (int i=0;i<hmat_0.nb_cols();i++){
            perm1[hmat_0.get_permt(i)]=i;
        }

        HPDDM::Option& opt = *HPDDM::Option::get();
        nevi = opt.val("geneo_nu",2);
// std::cout << nevi << std::endl;
        // nevi = 2;
        int mynev = nevi;
        evi.resize(nevi*n,0);
        // fill_n(evi.data(),n_inside,1);
        // std::copy_n(vr.data()+(n-n_inside)*n,nevi*n-(n-n_ins),evi.data());
        for (int i=0;i<nevi;i++){
            std::copy_n(vr.data()+(n-n_inside)*n+n*i,n_inside,evi.data()+i*n);
        }


        std::vector<T> buffer(nevi*n_global,0);
        std::vector<T> AZ(nevi*n_inside,0);
        E.resize(nevi*sizeWorld*sizeWorld*nevi,0);

        for (int i=0;i<sizeWorld;i++){
            std::fill_n(buffer.data(),buffer.size(),0);
            std::fill_n(AZ.data(),AZ.size(),0);

            if (i==rankWorld){
                for (int j=0;j<nevi;j++){
                    // std::copy_n(vr.data()+j*n,n_inside,buffer.data()+j*n_global+hmat_0.get_local_offset());
                    for (int k=0;k<n;k++){
                        buffer[nevi*perm1[renum_to_global[k]]+j]=evi[j*n+k]; // On décale du noyau
                    }
                }
            }

            MPI_Bcast(&nevi,1,MPI_INT,i,comm);
            MPI_Bcast(buffer.data(),nevi*n_global,wrapper_mpi<T>::mpi_type(),i,comm);
            hmat_0.mymvprod_local(buffer.data(),AZ.data(),nevi);
            // if (rankWorld==0){
            //     std::cout << i << std::endl;
            //     for (int j=0;j<buffer.size();j++){
            //         std::cout << j<<" "<<buffer[j] << std::endl;
            //     }
            // }
            // std::copy_n(buffer.data(),nevi*n_global,AZ.data());
            // for (int j=0;j<n_inside;j++){
            //     std::copy_n(buffer.data()+(hmat_0.get_local_offset()+j)*nevi,nevi,AZ.data()+j*nevi);
            // }
            for (int j=0;j<nevi;j++){
                // std::fill_n(vec_ovr.data(),vec_ovr.size(),0);
                // std::copy_n(AZ.data()+j*n_inside+hmat_0.get_local_offset(),n_inside,vec_ovr.data());
                for (int k=0;k<n_inside;k++){
                    vec_ovr[k]=AZ[j+nevi*k];
                }
                synchronize(true);
                for (int jj=0;jj<mynev;jj++){
                    int coord_E_i = nevi*i+j;
                    int coord_E_j = nevi*rankWorld+jj;
                    E[coord_E_i+coord_E_j*nevi*sizeWorld]=std::inner_product(evi.data()+jj*n,evi.data()+jj*n+n,vec_ovr.data(),T(0),std::plus<T >(), [](T u,T v){return u*std::conj(v);});
                }

            }
        }
        if (rankWorld==0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);
        else
            MPI_Reduce(E.data(), E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);

        // if (rankWorld==0){
        //     std::cout << E << std::endl;
        // }
        mytime[3] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        int n_coarse = nevi*sizeWorld;
        _ipiv_coarse.resize(E.size());

        HPDDM::Lapack<T>::getrf(&n_coarse,&n_coarse,E.data(),&n_coarse,_ipiv_coarse.data(),&info);
        mytime[4] = MPI_Wtime() - time;
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 5, MPI_DOUBLE, MPI_MAX, 0,this->comm);
        MPI_Reduce(&(mytime[0]), &(meantime[0]), 5, MPI_DOUBLE, MPI_SUM, 0,this->comm);
        meantime /= hmat_0.get_sizeworld();

        infos["DDM_setup_one_level_mean"]= NbrToStr(meantime[0]);
        infos["DDM_setup_one_level_max" ]= NbrToStr(maxtime[0]);
        infos["DDM_facto_one_level_mean"]= NbrToStr(meantime[1]);
        infos["DDM_facto_one_level_max" ]= NbrToStr(maxtime[1]);
        infos["DDM_geev_mean"]= NbrToStr(meantime[2]);
        infos["DDM_geev_max" ]= NbrToStr(maxtime[2]);
        infos["DDM_setup_ZtAZ_mean"]= NbrToStr(meantime[3]);
        infos["DDM_setup_ZtAZ_max" ]= NbrToStr(maxtime[3]);
        infos["DDM_facto_ZtAZ_mean"]= NbrToStr(meantime[4]);
        infos["DDM_facto_ZtAZ_max" ]= NbrToStr(maxtime[4]);
    }


    void one_level(const T* const in, T* const out){
        // Timing
        double mytime, maxtime, meantime;
        double time = MPI_Wtime();
        int sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);

        // Without overlap to with overlap
        std::copy_n(in,n_inside,vec_ovr.data());
        // std::cout << n<<" "<<n_inside <<std::endl;
        // std::fill(vec_ovr.begin(),vec_ovr.end(),0);
        // std::fill_n(vec_ovr.begin(),n_inside,1);
        synchronize(true);
        // std::cout << n_inside<<std::endl;
        // std::cout << n<<std::endl;
        const char l='N';
        int lda=n;
        int ldb=n;
        int nrhs =1 ;
        int info;
        // std::cout << n <<" "<<n_inside<<" "<<mat_loc.size()<<" "<<vec_ovr.size()<<std::endl;
        HPDDM::Lapack<T>::getrs(&l,&n,&nrhs,mat_loc.data(),&lda,_ipiv.data(),vec_ovr.data(),&ldb,&info);

// std::cout << info << std::endl;
        HPDDM::Option& opt = *HPDDM::Option::get();
        int schwarz_method = opt.val("schwarz_method",HPDDM_SCHWARZ_METHOD_RAS);
        if (schwarz_method==HPDDM_SCHWARZ_METHOD_RAS){
            synchronize(true);
        }
        else if (schwarz_method==HPDDM_SCHWARZ_METHOD_ASM) {
            synchronize(false);
        }

        std::copy_n(vec_ovr.data(),n_inside,out);

        timing_one_level += MPI_Wtime() - time;

    }

    void Q(const T* const in, T* const out){
        // Timing
        double mytime, maxtime, meantime;
        double time = MPI_Wtime();

        std::copy_n(in,n_inside,vec_ovr.data());
        synchronize(true);
        std::vector<T> zti(nevi);
        int sizeWorld;
        int rankWorld;
        MPI_Comm_size(comm,&sizeWorld);
        MPI_Comm_rank(comm,&rankWorld);

        for (int i=0;i<nevi;i++){
            zti[i]=std::inner_product(evi.begin()+i*n,evi.begin()+i*n+n,vec_ovr.begin(),T(0),std::plus<T>(), [](T u,T v){return u*std::conj(v);});
        }
        std::vector<T> zt(nevi*sizeWorld,0);
        // if (rankWorld==0){
        //     // std::cout << zt.size() <<" " << zti.size() << std::endl;
        // }
        MPI_Gather(zti.data(),zti.size(),wrapper_mpi<T>::mpi_type(),zt.data(),zti.size(),wrapper_mpi<T>::mpi_type(),0,comm);

        if (rankWorld==0){
            const char l='N';
            int zt_size = zt.size();
            int lda=zt_size;
            int ldb=zt_size;
            int nrhs =1 ;
            int info;
            // std::cout << n <<" "<<n_inside<<" "<<mat_loc.size()<<" "<<vec_ovr.size()<<std::endl;
            HPDDM::Lapack<T>::getrs(&l,&zt_size,&nrhs,E.data(),&lda,_ipiv_coarse.data(),zt.data(),&ldb,&info);
        }

        MPI_Scatter(zt.data(),zti.size(),wrapper_mpi<T>::mpi_type(),zti.data(),zti.size(),wrapper_mpi<T>::mpi_type(),0,comm);

        std::fill_n(vec_ovr.data(),n,0);
        for (int i=0;i<nevi;i++){

            std::transform(vec_ovr.begin(),vec_ovr.begin()+n,evi.begin()+n*i,vec_ovr.begin(),[&i,&zti](T u, T v){return u+v*zti[i];});
        }
        synchronize(true);
        std::copy_n(vec_ovr.data(),n_inside,out);

        timing_Q += MPI_Wtime() - time;

    }

    void apply(const T* const in, T* const out){
        std::vector<T> out_one_level(n_inside,0);
        std::vector<T> out_Q(n_inside,0);
        std::vector<T> buffer(hmat.nb_cols());
        std::vector<T> aq(n_inside);
        std::vector<T> p(n_inside);
        std::vector<T> am1p(n_inside);
        std::vector<T> qam1p(n_inside);
        std::vector<T> ptm1p(n_inside);

        HPDDM::Option& opt = *HPDDM::Option::get();
        int schwarz_method = opt.val("schwarz_method",HPDDM_SCHWARZ_METHOD_RAS);
        if (schwarz_method==HPDDM_SCHWARZ_METHOD_NONE){
            std::copy_n(in,n_inside,out);
        }
        else{
            switch (opt.val("schwarz_coarse_correction",42)) {
                case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                    Q(in,out_Q.data());
                    hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
                    std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());

                    one_level(p.data(),out_one_level.data());
                    hmat.mvprod_local(out_one_level.data(),am1p.data(),buffer.data(),1);
                    Q(am1p.data(),qam1p.data());

                    std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,qam1p.begin(),ptm1p.data(),std::minus<T>());
                    std::transform(out_Q.begin(),out_Q.begin()+n_inside,ptm1p.begin(),out,std::plus<T>());
                    break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                    Q(in,out_Q.data());
                    hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
                    std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());
                    one_level(p.data(),out_one_level.data());
                    std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());
                    break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                    one_level(in,out_one_level.data());
                    Q(in,out_Q.data());
                    std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());
                    break;
                default:
                    one_level(in,out);
                    break;

            }
        }
        // ASM
        // one_level(in,out);
        // Q(in,out_Q.data());
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());


        // // ADEF1
        // Q(in,out_Q.data());
        // std::vector<T> aq(n_inside);
        // hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
        // std::vector<T> p(n_inside);
        //
        // std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());
        //
        // one_level(p.data(),out_one_level.data());
        //
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());

        // // BNN
        //
        // Q(in,out_Q.data());
        // hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
        // std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());
        //
        // one_level(p.data(),out_one_level.data());
        // hmat.mvprod_local(out_one_level.data(),am1p.data(),buffer.data(),1);
        // Q(am1p.data(),qam1p.data());
        //
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,qam1p.begin(),ptm1p.data(),std::minus<T>());
        // std::transform(out_Q.begin(),out_Q.begin()+n_inside,ptm1p.begin(),out,std::plus<T>());
    }

    int get_n() const {return n;}
    int get_n_inside() const {return n_inside;}
    std::map<std::string, std::string>& get_infos() const{return infos;}
    double get_timing_one_level() const {return timing_one_level;}
    double get_timing_Q() const {return timing_Q;}


};

}
#endif
