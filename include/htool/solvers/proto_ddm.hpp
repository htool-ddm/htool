#ifndef HTOOL_PROTO_DDM_HPP
#define HTOOL_PROTO_DDM_HPP

#include "../types/matrix.hpp"
#include "../types/hmatrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../wrappers/wrapper_hpddm.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class Proto_HPDDM;

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
    std::vector<int> renum_to_global;
    std::vector<T> E;
    // Matrix<T>& evp;
    const HMatrix<LowRankMatrix,T>& hmat;
    const T* const* Z;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    int size_E;


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

    double timing_one_level;
    double timing_Q;

    Proto_DDM(const IMatrix<T>& mat0, const HMatrix<LowRankMatrix,T>& hmat_0,
    const std::vector<int>&  ovr_subdomain_to_global0,
    const std::vector<int>& cluster_to_ovr_subdomain0,
    const std::vector<int>& neighbors0,
    const std::vector<std::vector<int> >& intersections0):  n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n),mat_loc(n*n), D(n), comm(hmat_0.get_comm()), _ipiv(n), hmat(hmat_0), timing_Q(0), timing_one_level(0), recvcounts(hmat_0.get_sizeworld()), displs(hmat_0.get_sizeworld()) {

        // Timing
        MPI_Barrier(hmat.get_comm());
        double mytime, maxtime;
        double time = MPI_Wtime();




        std::vector<int> renum(n,-1);
        renum_to_global.resize(n);

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


        // Timing
        mytime =  MPI_Wtime() - time;
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_setup_one_level_max" ]= NbrToStr(maxtime);
        // infos["DDM_facto_one_level_max" ]= NbrToStr(maxtime[1]);


        //

        snd.resize(neighbors.size());
        rcv.resize(neighbors.size());

        for (int i=0;i<neighbors.size();i++){
          snd[i].resize(intersections[i].size());
          rcv[i].resize(intersections[i].size());
        }
    }

    void facto_one_level(){
        double time = MPI_Wtime();
        double mytime, maxtime;
        int lda=n;
        int info;

        HPDDM::Lapack<T>::getrf(&n,&n,mat_loc.data(),&lda,_ipiv.data(),&info);

        mytime = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());


        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_facto_one_level_max" ]= NbrToStr(maxtime);
    }

    void build_coarse_space( Matrix<T>& Mi, Matrix<T>& Bi){
        // Timing
        std::vector<double> mytime(4), maxtime(4);
        double time = MPI_Wtime();
        // Data
        int n_global= hmat.nb_cols();
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info;

        // LU facto for mass matrix
        int lda=n;
        std::vector<int> _ipiv_mass(n);
        HPDDM::Lapack<Cplx>::getrf(&n,&n,Mi.data(),&lda,_ipiv_mass.data(),&info);

        // Partition of unity
        Matrix<T> DAiD(n,n);
        for (int i =0 ;i < n_inside;i++){
            std::copy_n(&(mat_loc[i*n]),n_inside,&(DAiD(0,i)));
        }

        // M^-1
        const char l='N';
        lda=n;
        int ldb=n;
        HPDDM::Lapack<Cplx>::getrs(&l,&n,&n,Mi.data(),&lda,_ipiv_mass.data(),DAiD.data(),&ldb,&info);
        HPDDM::Lapack<Cplx>::getrs(&l,&n,&n,Mi.data(),&lda,_ipiv_mass.data(),Bi.data(),&ldb,&info);

        // Build local eigenvalue problem
        Matrix<T> evp(n,n);
        Bi.mvprod(DAiD.data(),evp.data(),n);

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Local eigenvalue problem
        int ldvl = n, ldvr = n, lwork=-1;
        lda=n;
        std::vector<T> work(n);
        std::vector<double> rwork(2*n);
        std::vector<T> w(n);
        std::vector<T> vl(n*n), vr(n*n);

        HPDDM::Lapack<T>::geev( "N", "Vectors", &n, evp.data(), &lda, w.data(),nullptr , vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::geev( "N", "Vectors", &n, evp.data(), &lda, w.data(),nullptr , vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );
        std::vector<int> index(n, 0);

        for (int i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(),
            [&](const int& a, const int& b) {
                return (std::abs(w[a]) > std::abs(w[b]));
            }
        );
        HPDDM::Option& opt = *HPDDM::Option::get();
        nevi=0;
        double threshold = opt.val("geneo_threshold",-1.0);
        if (threshold > 0.0){
            while (std::abs(w[index[nevi]])>threshold && nevi< index.size()){
                nevi++;}


        }
        else {
            nevi = opt.val("geneo_nu",2);
        }
        evi.resize(nevi*n);


        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Allgather
        MPI_Allgather(&nevi,1,MPI_INT,recvcounts.data(),1,MPI_INT,comm);

        displs[0] = 0;

        for (int i=1; i<sizeWorld; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        //
        size_E   =  std::accumulate(recvcounts.begin(),recvcounts.end(),0);
        int nevi_max = *std::max_element(recvcounts.begin(),recvcounts.end());
        evi.resize(nevi*n,0);
        for (int i=0;i<nevi;i++){
            // std::fill_n(evi.data()+i*n,n_inside,rankWorld+1);
            // std::copy_n(Z[i],n_inside,evi.data()+i*n);
            std::copy_n(vr.data()+index[i]*n,n_inside,evi.data()+i*n);
        }
        // for (int i=0;i<n;i++){
        //     for (int j=0;j<nevi;j++){
        //         evi[i+j*nevi]/= std::sqrt(norms[j]);
        //     }
        // }

        // if (rankWorld==0){
        //     for (int i=0;i<nevi;i++){
        //         for (int j=0;j<n;j++){
        //             // std::cout << Z[i][j]<<" ";
        //             std::cout << vr[index[i]*n+j]<<" ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }


        // for (int i=0;i<nevi;i++){
        //     double norm_i = 0;
        //     for (int j=0;j<n;j++){
        //         norm_i += std::sqrt(std::abs(evi[i*n+j]*std::conj(evi[i*n+j])));
        //     }
        //     for (int j=0;j<n;j++){
        //         evi[i*n+j] /= norm_i;
        //     }
        // }

        // for (int i=0;i<sizeWorld;i++){
        //     MPI_Barrier(comm);
        //     if (rankWorld==i){
        //         std::cout << "proc "<<i<< std::endl;
        //         std::cout << evi << std::endl;
        //     }
        //     MPI_Barrier(comm);
        // }
        // MPI_Barrier(comm);

        // if (rankWorld==0){
        //     for (int i=0;i<nevi;i++){
        //         std::cout << w[index[i]]<<" ";
        //     }
        //     std::cout << std::endl;
        // }

        std::vector<T> AZ(nevi_max*n_inside,0);
        E.resize(size_E*size_E,0);

        for (int i=0;i<sizeWorld;i++){
            std::vector<T> buffer(hmat.get_MasterOffset_t(i).second*recvcounts[i],0);
            std::fill_n(AZ.data(),recvcounts[i]*n_inside,0);

            if (rankWorld==i){
                for (int j=0;j<recvcounts[i];j++){
                    for (int k=0;k<n_inside;k++){
                        buffer[recvcounts[i]*k+j]=evi[j*n+k];
                    }
                }
            }
            MPI_Bcast(buffer.data(),hmat.get_MasterOffset_t(i).second*recvcounts[i],wrapper_mpi<T>::mpi_type(),i,comm);


            // if (i==rankWorld){
            //     for (int j=0;j<nevi;j++){
            //         // std::copy_n(vr.data()+j*n,n_inside,buffer.data()+j*n_global+hmat_0.get_local_offset());
            //         for (int k=0;k<n;k++){
            //             buffer[nevi*perm1[renum_to_global[k]]+j]=evi[j*n+k];
            //             // buffer[nevi*perm1[renum_to_global[k]]+j]=Z[j][k];
            //         }
            //     }
            //     // std::cout << "proc "<<rankWorld<< std::endl;
            //     // std::cout << buffer << std::endl;
            // }



            // MPI_Barrier(comm);
            // if (rankWorld==0){
            //     // std::cout << "avant "<< std::endl;
            //     std::cout << buffer << std::endl;
            // }

            // MPI_Bcast(buffer.data(),nevi*n_global,wrapper_mpi<T>::mpi_type(),i,comm);
            hmat.mvprod_subrhs(buffer.data(),AZ.data(),recvcounts[i],hmat.get_MasterOffset_t(i).first,hmat.get_MasterOffset_t(i).second);
            // if (rankWorld==0){
            //     std::cout << recvcounts[i]<< std::endl;
            //     std::cout << AZ << std::endl;
            // }
            // if (rankWorld==0){
            //     std::cout << "après "<< std::endl;
            //     std::cout << buffer << std::endl;
            // }
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
            for (int j=0;j<recvcounts[i];j++){
                // std::fill_n(vec_ovr.data(),vec_ovr.size(),0);
                // std::copy_n(AZ.data()+j*n_inside+hmat_0.get_local_offset(),n_inside,vec_ovr.data());
                for (int k=0;k<n_inside;k++){
                    vec_ovr[k]=AZ[j+recvcounts[i]*k];
                }
                // Parce que partition de l'unité...
                // synchronize(true);
                for (int jj=0;jj<nevi;jj++){
                    int coord_E_i = displs[i]+j;
                    int coord_E_j = displs[rankWorld]+jj;
                    E[coord_E_i+coord_E_j*size_E]=std::inner_product(evi.data()+jj*n,evi.data()+jj*n+n_inside,vec_ovr.data(),T(0),std::plus<T >(), [](T u,T v){return u*std::conj(v);});
                    // E[coord_E_i+coord_E_j*nevi*sizeWorld]=std::inner_product(Z[jj],Z[jj+1],vec_ovr.data(),T(0),std::plus<T >(), [](T u,T v){return u*std::conj(v);});

                }
            }
        }
        if (rankWorld==0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);
        else
            MPI_Reduce(E.data(), E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);

        // if (rankWorld==0){
        //     std::cout << "size E :"<<E.size() << std::endl;
        //     for (int i=0;i<nevi*sizeWorld;i++){
        //         for (int j=0;j<nevi*sizeWorld;j++){
        //             std::cout << E[i+j*nevi*sizeWorld] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }
        mytime[2] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        int n_coarse = size_E;
        _ipiv_coarse.resize(n_coarse);

        HPDDM::Lapack<T>::getrf(&n_coarse,&n_coarse,E.data(),&n_coarse,_ipiv_coarse.data(),&info);
        mytime[3] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 4, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_setup_geev_max" ]= NbrToStr(maxtime[0]);
        infos["DDM_geev_max" ]= NbrToStr(maxtime[1]);
        infos["DDM_setup_ZtAZ_max" ]= NbrToStr(maxtime[2]);
        infos["DDM_facto_ZtAZ_max" ]= NbrToStr(maxtime[3]);
    }


    void one_level(const T* const in, T* const out){
        int sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);

        // Without overlap to with overlap
        std::copy_n(in,n_inside,vec_ovr.data());
        // std::cout << n<<" "<<n_inside <<std::endl;
        // std::fill(vec_ovr.begin(),vec_ovr.end(),0);
        // std::fill_n(vec_ovr.begin(),n_inside,1);
        synchronize(true);

        // Timing
        MPI_Barrier(hmat.get_comm());
        double time = MPI_Wtime();


        // std::cout << n_inside<<std::endl;
        // std::cout << n<<std::endl;
        const char l='N';
        int lda=n;
        int ldb=n;
        int nrhs =1 ;
        int info;
        // std::cout << n <<" "<<n_inside<<" "<<mat_loc.size()<<" "<<vec_ovr.size()<<std::endl;
        HPDDM::Lapack<T>::getrs(&l,&n,&nrhs,mat_loc.data(),&lda,_ipiv.data(),vec_ovr.data(),&ldb,&info);



        timing_one_level += MPI_Wtime() - time;

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


    }

    void Q(const T* const in, T* const out){
        std::copy_n(in,n_inside,vec_ovr.data());
        synchronize(true);
        std::vector<T> zti(nevi);
        int sizeWorld;
        int rankWorld;
        MPI_Comm_size(comm,&sizeWorld);
        MPI_Comm_rank(comm,&rankWorld);


        // Timing
        MPI_Barrier(hmat.get_comm());
        double time = MPI_Wtime();

        for (int i=0;i<nevi;i++){
            zti[i]=std::inner_product(evi.begin()+i*n,evi.begin()+i*n+n,vec_ovr.begin(),T(0),std::plus<T>(), [](T u,T v){return u*std::conj(v);});
            // zti[i]=std::inner_product(Z[i],Z[i+1],vec_ovr.begin(),T(0),std::plus<T>(), [](T u,T v){return u*std::conj(v);});
        }
        std::vector<T> zt(size_E,0);
        // if (rankWorld==0){
        //     // std::cout << zt.size() <<" " << zti.size() << std::endl;
        // }
        MPI_Gatherv(zti.data(),zti.size(),wrapper_mpi<T>::mpi_type(),zt.data(),recvcounts.data(),displs.data(),wrapper_mpi<T>::mpi_type(),0,comm);

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

        MPI_Scatterv(zt.data(),recvcounts.data(),displs.data(),wrapper_mpi<T>::mpi_type(),zti.data(),zti.size(),wrapper_mpi<T>::mpi_type(),0,comm);

        std::fill_n(vec_ovr.data(),n,0);
        for (int i=0;i<nevi;i++){

            std::transform(vec_ovr.begin(),vec_ovr.begin()+n,evi.begin()+n*i,vec_ovr.begin(),[&i,&zti](T u, T v){return u+v*zti[i];});
            // std::transform(vec_ovr.begin(),vec_ovr.begin()+n,Z[i],vec_ovr.begin(),[&i,&zti](T u, T v){return u+v*zti[i];});
        }

        timing_Q += MPI_Wtime() - time;

        synchronize(true);
        std::copy_n(vec_ovr.data(),n_inside,out);



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

    void init_hpddm(Proto_HPDDM<LowRankMatrix,T>& hpddm_op){
        bool sym=false;
        hpddm_op.initialize(n, sym, nullptr, neighbors, intersections);
    }
    int get_n() const {return n;}
    int get_n_inside() const {return n_inside;}
    int get_size_E() const {return size_E;}
    std::map<std::string, std::string>& get_infos() const{return infos;}
    double get_timing_one_level() const {return timing_one_level;}
    double get_timing_Q() const {return timing_Q;}


};

}
#endif
