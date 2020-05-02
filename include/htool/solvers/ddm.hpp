#ifndef HTOOL_DDM_HPP
#define HTOOL_DDM_HPP

#include "../types/matrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../wrappers/wrapper_hpddm.hpp"

namespace htool{

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
class DDM{
private:
    int n;
    int n_inside;
    const std::vector<int> neighbors;
    std::vector<int> renum_to_global;
    // const std::vector<int> cluster_to_ovr_subdomain;
    std::vector<std::vector<int> > intersections;
    std::vector<T> vec_ovr;
    HPDDMDense<T,LowRankMatrix,ClusterImpl> hpddm_op;
    std::vector<T> mat_loc;
    std::vector<double> D;
    const MPI_Comm& comm;
    int nevi;
    int size_E;
    bool one_level;
    bool two_level;
    mutable std::map<std::string, std::string> infos;

    T** Z;


public:

    void clean(){
        hpddm_op.~HPDDMDense<T,LowRankMatrix,ClusterImpl>();
    }

    // Without overlap
    DDM(const HMatrix<T,LowRankMatrix,ClusterImpl>& hmat_0):n(hmat_0.get_local_size()),n_inside(hmat_0.get_local_size()),hpddm_op(hmat_0),mat_loc(n*n),D(n),nevi(0),size_E(0),comm(hmat_0.get_comm()),one_level(0),two_level(0){
        // Timing
        double mytime, maxtime, meantime;
        double time = MPI_Wtime();

        // Building Ai
        bool sym=false;
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& MyDiagFarFieldMats = hpddm_op.HA.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyDiagNearFieldMats= hpddm_op.HA.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int l=0;l<MyDiagNearFieldMats.size();l++){
            const SubMatrix<T>& submat = *(MyDiagNearFieldMats[l]);
            int local_nr = submat.nb_rows();
            int local_nc = submat.nb_cols();
            int offset_i = submat.get_offset_i()-hpddm_op.HA.get_local_offset();;
            int offset_j = submat.get_offset_j()-hpddm_op.HA.get_local_offset();
            for (int k=0;k<local_nc;k++){
                std::copy_n(&(submat(0,k)),local_nr,&mat_loc[offset_i+(offset_j+k)*n]);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n,n);
        for (int l=0;l<MyDiagFarFieldMats.size();l++){
            const LowRankMatrix<T,ClusterImpl>& lmat = *(MyDiagFarFieldMats[l]);
            int local_nr = lmat.nb_rows();
            int local_nc = lmat.nb_cols();
            int offset_i = lmat.get_offset_i()-hpddm_op.HA.get_local_offset();
            int offset_j = lmat.get_offset_j()-hpddm_op.HA.get_local_offset();;
            FarFielBlock.resize(local_nr,local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0,0)));
            for (int k=0;k<local_nc;k++){
                std::copy_n(&(FarFielBlock(0,k)),local_nr,&mat_loc[offset_i+(offset_j+k)*n]);
            }
        }

        std::vector<int> neighbors;
        std::vector<std::vector<int> > intersections;
        hpddm_op.initialize(n, sym, mat_loc.data(), neighbors, intersections);

        fill(D.begin(),D.begin()+n_inside,1);
        fill(D.begin()+n_inside,D.end(),0);

        hpddm_op.HPDDMDense<T,LowRankMatrix,ClusterImpl>::super::super::initialize(D.data());
        mytime =  MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_setup_one_level_max" ]= NbrToStr(maxtime);
    }

    // With overlap
    DDM(const IMatrix<T>& mat0, const HMatrix<T,LowRankMatrix,ClusterImpl>& hmat_0,
    const std::vector<int>&  ovr_subdomain_to_global0,
    const std::vector<int>& cluster_to_ovr_subdomain0,
    const std::vector<int>& neighbors0,
    const std::vector<std::vector<int> >& intersections0): hpddm_op(hmat_0), n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n),mat_loc(n*n), D(n), comm(hmat_0.get_comm()),one_level(0),two_level(0) {

        // Timing
        double mytime, maxtime, meantime;
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
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& MyDiagFarFieldMats = hpddm_op.HA.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyDiagNearFieldMats= hpddm_op.HA.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int l=0;l<MyDiagNearFieldMats.size();l++){
          const SubMatrix<T>& submat = *(MyDiagNearFieldMats[l]);
          int local_nr = submat.nb_rows();
          int local_nc = submat.nb_cols();
          int offset_i = submat.get_offset_i()-hpddm_op.HA.get_local_offset();;
          int offset_j = submat.get_offset_j()-hpddm_op.HA.get_local_offset();
          for (int k=0;k<local_nc;k++){
            std::copy_n(&(submat(0,k)),local_nr,&mat_loc[offset_i+(offset_j+k)*n]);
          }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n,n);
        for (int l=0;l<MyDiagFarFieldMats.size();l++){
          const LowRankMatrix<T,ClusterImpl>& lmat = *(MyDiagFarFieldMats[l]);
          int local_nr = lmat.nb_rows();
          int local_nc = lmat.nb_cols();
          int offset_i = lmat.get_offset_i()-hpddm_op.HA.get_local_offset();
          int offset_j = lmat.get_offset_j()-hpddm_op.HA.get_local_offset();;
          FarFielBlock.resize(local_nr,local_nc);
          lmat.get_whole_matrix(&(FarFielBlock(0,0)));
          for (int k=0;k<local_nc;k++){
            std::copy_n(&(FarFielBlock(0,k)),local_nr,&mat_loc[offset_i+(offset_j+k)*n]);
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

        hpddm_op.initialize(n, sym, mat_loc.data(), neighbors, intersections);

        fill(D.begin(),D.begin()+n_inside,1);
        fill(D.begin()+n_inside,D.end(),0);

        hpddm_op.HPDDMDense<T,LowRankMatrix,ClusterImpl>::super::super::initialize(D.data());
        mytime =  MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0,comm);

        infos["DDM_setup_one_level_max" ]= NbrToStr(maxtime);

    }

    void facto_one_level(){
        double time = MPI_Wtime();
        double mytime, maxtime;
        hpddm_op.callNumfact();
        mytime = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_facto_one_level_max" ]= NbrToStr(maxtime);
        one_level=1;
    }

    void build_coarse_space( Matrix<T>& Mi, IMatrix<T>& generator_Bi, const std::vector<R3>& x ){

        // Timing
        std::vector<double> mytime(4), maxtime(4);
        double time = MPI_Wtime();

        //
        int n_global= hpddm_op.HA.nb_cols();
        int sizeWorld = hpddm_op.HA.get_sizeworld();
        int rankWorld = hpddm_op.HA.get_rankworld();
        int info;

        // Building Neumann matrix
        htool::HMatrix<T,LowRankMatrix,ClusterImpl> HBi(generator_Bi,hpddm_op.HA.get_cluster_tree_t().get_local_cluster_tree(),x,-1,MPI_COMM_SELF);
        Matrix<T> Bi(n,n);

        // Building Bi
        bool sym=false;
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& MyLocalFarFieldMats = HBi.get_MyFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyLocalNearFieldMats= HBi.get_MyNearFieldMats();
        // std::cout << MyLocalNearFieldMats.size()<<std::endl;
        // std::cout << MyLocalFarFieldMats.size()<<std::endl;

        // Internal dense blocks
        for (int i=0;i<MyLocalNearFieldMats.size();i++){
          const SubMatrix<T>& submat = *(MyLocalNearFieldMats[i]);
          int local_nr = submat.nb_rows();
          int local_nc = submat.nb_cols();
          int offset_i = submat.get_offset_i()-hpddm_op.HA.get_local_offset();
          int offset_j = submat.get_offset_j()-hpddm_op.HA.get_local_offset();
          for (int i=0;i<local_nc;i++){
            std::copy_n(&(submat(0,i)),local_nr,Bi.data()+offset_i+(offset_j+i)*n);
          }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n,n);
        for (int i=0;i<MyLocalFarFieldMats.size();i++){
          const LowRankMatrix<T,ClusterImpl>& lmat = *(MyLocalFarFieldMats[i]);
          int local_nr = lmat.nb_rows();
          int local_nc = lmat.nb_cols();
          int offset_i = lmat.get_offset_i()-hpddm_op.HA.get_local_offset();
          int offset_j = lmat.get_offset_j()-hpddm_op.HA.get_local_offset();;
          FarFielBlock.resize(local_nr,local_nc);
          lmat.get_whole_matrix(&(FarFielBlock(0,0)));
          for (int i=0;i<local_nc;i++){
            std::copy_n(&(FarFielBlock(0,i)),local_nr,Bi.data()+offset_i+(offset_j+i)*n);
          }
        }


        // Overlap
        std::vector<T> horizontal_block(n-n_inside,n_inside),vertical_block(n,n-n_inside);
        horizontal_block = generator_Bi.get_submatrix(std::vector<int>(renum_to_global.begin()+n_inside,renum_to_global.end()),std::vector<int>(renum_to_global.begin(),renum_to_global.begin()+n_inside)).get_mat();
        vertical_block = generator_Bi.get_submatrix(renum_to_global,std::vector<int>(renum_to_global.begin()+n_inside,renum_to_global.end())).get_mat();
        for (int j=0;j<n_inside;j++){
          std::copy_n(horizontal_block.begin()+j*(n-n_inside),n-n_inside,Bi.data()+n_inside+j*n);
        }
        for (int j=n_inside;j<n;j++){
          std::copy_n(vertical_block.begin()+(j-n_inside)*n,n,Bi.data()+j*n);
        }

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
        MPI_Barrier(hpddm_op.HA.get_comm());
        time = MPI_Wtime();

        // if (rankWorld==0)
        //     evp.matlab_save("evp_ddm.txt");

        // eigenvalue problem
        hpddm_op.solveEVP(evp.data());
        T* const* Z = const_cast <T* const*> (hpddm_op.getVectors()) ;
        HPDDM::Option& opt = *HPDDM::Option::get();
        nevi = opt.val("geneo_nu",2);

        // for (int i=0;i<nevi;i++){
        //     double norm_i=0;
        //     for (int j=0;j<n;j++){
        //         norm_i+=std::abs(Z[i][j]*std::conj(Z[i][j]));
        //     }
        //     norm_i=std::sqrt(norm_i);
        //     for (int j=0;j<n;j++){
        //         Z[i][j]=Z[i][j]/norm_i;
        //     }
        //
        // }

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op.HA.get_comm());
        time = MPI_Wtime();


        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);
        // std::cout << rankWorld << " " <<nevi <<std::endl;
        MPI_Allgather(&nevi,1,MPI_INT,recvcounts.data(),1,MPI_INT,comm);

        displs[0] = 0;

        for (int i=1; i<sizeWorld; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }


        size_E   =  std::accumulate(recvcounts.begin(),recvcounts.end(),0);
        int nevi_max = *std::max_element(recvcounts.begin(),recvcounts.end());
        std::vector<T >evi(nevi*n,0);
        for (int i=0;i<nevi;i++){
            // std::fill_n(evi.data()+i*n,n_inside,rankWorld+1);
            std::copy_n(Z[i],n_inside,evi.data()+i*n);
            // std::copy_n(vr.data()+index[i]*n,n_inside,evi.data()+i*n);
        }

        // Matrix<T> out_test(n,nevi);
        // for (int i=0;i<nevi;i++){
        //     std::vector<T> transvase(n);
        //     std::copy_n(Z[i],n,transvase.data());
        //     double norme=norm2(transvase);
        //     for (int i=0;i<n;i++){
        //         transvase[i]=transvase[i]/norme;
        //     }
        //     out_test.set_col(i,transvase);
        // }
        // if (rankWorld==0)
        //     out_test.matlab_save("evi_ddm.txt");



        int local_max_size_j=0;
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats = hpddm_op.HA.get_MyFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyNearFieldMats= hpddm_op.HA.get_MyNearFieldMats();
        for (int i=0;i<MyFarFieldMats.size();i++){
            if (local_max_size_j<(*MyFarFieldMats[i]).nb_cols())
                local_max_size_j=(*MyFarFieldMats[i]).nb_cols();
        }
        for (int i=0;i<MyNearFieldMats.size();i++){
            if (local_max_size_j<(*MyNearFieldMats[i]).nb_cols())
                local_max_size_j=(*MyNearFieldMats[i]).nb_cols();
        }

        std::vector<T> AZ(nevi_max*n_inside,0);
        std::vector<T> E;
        E.resize(size_E*size_E,0);

        for (int i=0;i<sizeWorld;i++){
            std::vector<T> buffer((hpddm_op.HA.get_MasterOffset_t(i).second+2*local_max_size_j)*recvcounts[i],0);
            std::fill_n(AZ.data(),recvcounts[i]*n_inside,0);

            if (rankWorld==i){
                for (int j=0;j<recvcounts[i];j++){
                    for (int k=0;k<n_inside;k++){
                        buffer[recvcounts[i]*(k+local_max_size_j)+j]=evi[j*n+k];
                    }
                }
            }
            MPI_Bcast(buffer.data()+local_max_size_j*recvcounts[i],hpddm_op.HA.get_MasterOffset_t(i).second*recvcounts[i],wrapper_mpi<T>::mpi_type(),i,comm);


            hpddm_op.HA.mvprod_subrhs(buffer.data(),AZ.data(),recvcounts[i],hpddm_op.HA.get_MasterOffset_t(i).first,hpddm_op.HA.get_MasterOffset_t(i).second,local_max_size_j);

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

                }
            }
        }
        if (rankWorld==0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);
        else
            MPI_Reduce(E.data(), E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);
        // if (rankWorld==0){
        //     double norme=0;
        //     std::cout << "size E :"<<E.size() << std::endl;
        //     std::cout << "[";
        //     for (int i=0;i<nevi*sizeWorld;i++){
        //         std::cout << "[";
        //         for (int j=0;j<nevi*sizeWorld;j++){
        //             std::cout << std::real(E[i+j*nevi*sizeWorld])<<"+"<<std::imag(E[i+j*nevi*sizeWorld]) << "i,";
        //             norme+=std::abs(E[i+j*nevi*sizeWorld]*std::conj(E[i+j*nevi*sizeWorld]));
        //         }
        //         std::cout << "];";
        //     }
        //     std::cout << "]"<<std::endl;
        //     std::cout << "NORME : "<<norme<<std::endl;
        // }
        // if (rankWorld==0)
        //     matlab_save(E,"E_ddm.txt");
        mytime[2] = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op.HA.get_comm());
        time = MPI_Wtime();

        hpddm_op.buildTwo(MPI_COMM_WORLD, E.data());

        mytime[3] = MPI_Wtime() - time;
        // MPI_Barrier(hmat.get_comm());
        // time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 4, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_setup_geev_max" ]= NbrToStr(maxtime[0]);
        infos["DDM_geev_max" ]= NbrToStr(maxtime[1]);
        infos["DDM_setup_ZtAZ_max" ]= NbrToStr(maxtime[2]);
        infos["DDM_facto_ZtAZ_max" ]= NbrToStr(maxtime[3]);

        two_level = 1;
    }

void build_coarse_space( Matrix<T>& Ki, const std::vector<R3>& x ){

        // Timing
        std::vector<double> mytime(3), maxtime(3);
        double time = MPI_Wtime();

        //
        int n_global= hpddm_op.HA.nb_cols();
        int sizeWorld = hpddm_op.HA.get_sizeworld();
        int rankWorld = hpddm_op.HA.get_rankworld();
        int info;

        // Partition of unity
        Matrix<T> DAiD(n,n);
        for (int i =0 ;i < n_inside;i++){
            std::copy_n(&(mat_loc[i*n]),n_inside,&(DAiD(0,i)));
        }


        // Build local eigenvalue problem
        int ldvl = n, ldvr = n, lwork=-1;
        int lda=n,ldb=n;
        std::vector<T> alpha(n),beta(n);
        std::vector<T> work(n);
        std::vector<double> rwork(8*n);
        std::vector<T> vl(n*n), vr(n*n);
        std::vector<int> index(n, 0);

        HPDDM::Lapack<T>::ggev( "N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alpha.data(),nullptr ,beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::ggev( "N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alpha.data(),nullptr ,beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info );

        for (int i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(),
            [&](const int& a, const int& b) {
                return ( (std::abs(beta[a])<1e-15 || (std::abs(alpha[a]/beta[a]) > std::abs(alpha[b]/beta[b]))) &&  !(std::abs(beta[b])<1e-15) );
            }
        );

        HPDDM::Option& opt = *HPDDM::Option::get();
        nevi=0;
        double threshold = opt.val("geneo_threshold",-1.0);
        if (threshold > 0.0){
            while (std::abs(beta[index[nevi]])<1e-15 || (std::abs(alpha[index[nevi]]/beta[index[nevi]])>threshold && nevi< index.size())){
                nevi++;}


        }
        else {
            nevi = opt.val("geneo_nu",2);
        }
        
        opt["geneo_nu"]=nevi;
        Z = new T*[nevi];
        *Z = new T[nevi*n];
        for (int i=0;i<nevi;i++){
            Z[i] = *Z + i * n;
            std::copy_n(vr.data()+index[i]*n,n_inside,Z[i]);
            for (int j=n_inside;j<n;j++){
                
                Z[i][j]=0;
            }
        }

        hpddm_op.setVectors(Z);

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op.HA.get_comm());
        time = MPI_Wtime();


        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);
        MPI_Allgather(&nevi,1,MPI_INT,recvcounts.data(),1,MPI_INT,comm);

        displs[0] = 0;

        for (int i=1; i<sizeWorld; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }


        size_E   =  std::accumulate(recvcounts.begin(),recvcounts.end(),0);
        int nevi_max = *std::max_element(recvcounts.begin(),recvcounts.end());
        std::vector<T >evi(nevi*n,0);
        for (int i=0;i<nevi;i++){
            // std::fill_n(evi.data()+i*n,n_inside,rankWorld+1);
            // std::copy_n(Z[i],n_inside,evi.data()+i*n);
            std::copy_n(vr.data()+index[i]*n,n_inside,evi.data()+i*n);
        }

        int local_max_size_j=0;
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats = hpddm_op.HA.get_MyFarFieldMats();
        const std::vector<SubMatrix<T>*>& MyNearFieldMats= hpddm_op.HA.get_MyNearFieldMats();
        for (int i=0;i<MyFarFieldMats.size();i++){
            if (local_max_size_j<(*MyFarFieldMats[i]).nb_cols())
                local_max_size_j=(*MyFarFieldMats[i]).nb_cols();
        }
        for (int i=0;i<MyNearFieldMats.size();i++){
            if (local_max_size_j<(*MyNearFieldMats[i]).nb_cols())
                local_max_size_j=(*MyNearFieldMats[i]).nb_cols();
        }

        std::vector<T> AZ(nevi_max*n_inside,0);
        std::vector<T> E;
        E.resize(size_E*size_E,0);

        for (int i=0;i<sizeWorld;i++){
            std::vector<T> buffer((hpddm_op.HA.get_MasterOffset_t(i).second+2*local_max_size_j)*recvcounts[i],0);
            std::fill_n(AZ.data(),recvcounts[i]*n_inside,0);

            if (rankWorld==i){
                for (int j=0;j<recvcounts[i];j++){
                    for (int k=0;k<n_inside;k++){
                        buffer[recvcounts[i]*(k+local_max_size_j)+j]=evi[j*n+k];
                    }
                }
            }
            MPI_Bcast(buffer.data()+local_max_size_j*recvcounts[i],hpddm_op.HA.get_MasterOffset_t(i).second*recvcounts[i],wrapper_mpi<T>::mpi_type(),i,comm);


            hpddm_op.HA.mvprod_subrhs(buffer.data(),AZ.data(),recvcounts[i],hpddm_op.HA.get_MasterOffset_t(i).first,hpddm_op.HA.get_MasterOffset_t(i).second,local_max_size_j);

            for (int j=0;j<recvcounts[i];j++){
                for (int k=0;k<n_inside;k++){
                    vec_ovr[k]=AZ[j+recvcounts[i]*k];
                }
                // Parce que partition de l'unité...
                // synchronize(true);
                for (int jj=0;jj<nevi;jj++){
                    int coord_E_i = displs[i]+j;
                    int coord_E_j = displs[rankWorld]+jj;
                    E[coord_E_i+coord_E_j*size_E]=std::inner_product(evi.data()+jj*n,evi.data()+jj*n+n_inside,vec_ovr.data(),T(0),std::plus<T >(), [](T u,T v){return u*std::conj(v);});

                }
            }
        }
        if (rankWorld==0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);
        else
            MPI_Reduce(E.data(), E.data(), E.size(), wrapper_mpi<T>::mpi_type(),MPI_SUM, 0,comm);

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op.HA.get_comm());
        time = MPI_Wtime();

        hpddm_op.buildTwo(MPI_COMM_WORLD, E.data());

        mytime[2] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 3, MPI_DOUBLE, MPI_MAX, 0,this->comm);

        infos["DDM_geev_max" ]= NbrToStr(maxtime[0]);
        infos["DDM_setup_ZtAZ_max" ]= NbrToStr(maxtime[1]);
        infos["DDM_facto_ZtAZ_max" ]= NbrToStr(maxtime[2]);
        two_level =1;
    }

    void solve(const T* const rhs, T* const x, const int& mu=1 ){
        // Check facto
        if (!one_level && two_level){
            std::cout << "ERROR: FACTO FOR ONE LEVEL MISSING"<< std::endl;
            exit(1);
        }

        // Eventually change one-level type
        HPDDM::Option& opt = *HPDDM::Option::get();
        switch (opt.val("schwarz_method",0)) {
            case HPDDM_SCHWARZ_METHOD_NONE:
            hpddm_op.setType(HPDDMDense<T,LowRankMatrix,ClusterImpl>::Prcndtnr::NO);
            break;
            case HPDDM_SCHWARZ_METHOD_RAS:
            hpddm_op.setType(HPDDMDense<T,LowRankMatrix,ClusterImpl>::Prcndtnr::GE);
            break;
            case HPDDM_SCHWARZ_METHOD_ASM:
            hpddm_op.setType(HPDDMDense<T,LowRankMatrix,ClusterImpl>::Prcndtnr::SY);
            break;
            // case HPDDM_SCHWARZ_METHOD_OSM:
            // hpddm_op.setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
            // case HPDDM_SCHWARZ_METHOD_ORAS:
            // hpddm_op.setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
            // case HPDDM_SCHWARZ_METHOD_SORAS:
            // hpddm_op.setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
        }

        //
        int rankWorld = hpddm_op.HA.get_rankworld();
        int sizeWorld = hpddm_op.HA.get_sizeworld();
        int offset  = hpddm_op.HA.get_local_offset();
        int size    = hpddm_op.HA.get_local_size();
        int nb_cols = hpddm_op.HA.nb_cols();
        int nb_rows = hpddm_op.HA.nb_rows();
        int nb_vec_prod =  StrToNbr<int>(hpddm_op.HA.get_infos("nb_mat_vec_prod"));
        double time = MPI_Wtime();

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n*mu,0);
        std::vector<T> local_rhs(n*mu,0);
        hpddm_op.in_global->resize(nb_cols*(mu==1 ? 1 : 2*mu));
        hpddm_op.buffer->resize(n_inside*(mu==1 ? 1 : 2*mu));

        // TODO: blocking ?
        for (int i=0;i<mu;i++){
            // Permutation
            hpddm_op.HA.source_to_cluster_permutation(rhs+i*nb_cols,rhs_perm.data());

            std::copy_n(rhs_perm.begin()+offset,n_inside,local_rhs.begin()+i*n);
        }

        // TODO: avoid com here
        // for (int i=0;i<n-n_inside;i++){
        //   local_rhs[i]=rhs_perm[]
        // }
        hpddm_op.scaledexchange(local_rhs.data(), mu);

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(hpddm_op, local_rhs.data(), x_local.data(), mu,comm);

        // Delete the overlap (useful only when mu>1 and n!=n_inside)
        for (int i=0;i<mu;i++){
            std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
    	std::vector<int> recvcounts(sizeWorld);
    	std::vector<int>  displs(sizeWorld);

    	displs[0] = 0;

        for (int i=0; i<sizeWorld; i++) {
        recvcounts[i] = (hpddm_op.HA.get_MasterOffset_t(i).second)*mu;
        if (i > 0)
        	displs[i] = displs[i-1] + recvcounts[i-1];
        }

        MPI_Allgatherv(local_rhs.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), hpddm_op.in_global->data() + (mu==1 ? 0 : mu*nb_rows), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

        //

        for (int i=0 ;i<mu;i++){
        if (mu!=1){
            for (int j=0; j<sizeWorld;j++){
                std::copy_n(hpddm_op.in_global->data()+mu*nb_rows+displs[j]+i*recvcounts[j]/mu,recvcounts[j]/mu,hpddm_op.in_global->data()+i*nb_rows+displs[j]/mu);
            }
        }

        // Permutation
        hpddm_op.HA.cluster_to_target_permutation(hpddm_op.in_global->data()+i*nb_rows,x+i*nb_rows);
        }


        // Infos
        time = MPI_Wtime()-time;
        infos["Solve"] = NbrToStr(time);
        infos["Nb_it"] = NbrToStr(nb_it);
        infos["Nb_subdomains"] = NbrToStr(sizeWorld);
        infos["nb_mat_vec_prod"] = NbrToStr(StrToNbr<int>(hpddm_op.HA.get_infos("nb_mat_vec_prod"))-nb_vec_prod);
        infos["mean_time_mat_vec_prod"] = NbrToStr(StrToNbr<double>(hpddm_op.HA.get_infos("total_time_mat_vec_prod"))/StrToNbr<double>(hpddm_op.HA.get_infos("nb_mat_vec_prod")));
        switch (opt.val("schwarz_method",0)) {
            case HPDDM_SCHWARZ_METHOD_NONE:
            infos["Precond"] = "None";
            break;
            case HPDDM_SCHWARZ_METHOD_RAS:
            infos["Precond"] = "RAS";
            break;
            case HPDDM_SCHWARZ_METHOD_ASM:
            infos["Precond"] = "ASM";
            break;
            // case HPDDM_SCHWARZ_METHOD_OSM:
            // infos["Precond"] = "OSM";
            // break;
            // case HPDDM_SCHWARZ_METHOD_ORAS:
            // infos["Precond"] = "ORAS";
            // break;
            // case HPDDM_SCHWARZ_METHOD_SORAS:
            // infos["Precond"] = "SORAS";
            // break;
        }

        switch (opt.val("krylov_method",8)) {
            case HPDDM_KRYLOV_METHOD_GMRES:
            infos["krylov_method"] = "gmres";
            break;
            case HPDDM_KRYLOV_METHOD_BGMRES:
            infos["krylov_method"] = "bgmres";
            break;
            case HPDDM_KRYLOV_METHOD_CG:
            infos["krylov_method"] = "cg";
            break;
            case HPDDM_KRYLOV_METHOD_BCG:
            infos["krylov_method"] = "bcg";
            break;
            case HPDDM_KRYLOV_METHOD_GCRODR:
            infos["krylov_method"] = "gcrodr";
            break;
            case HPDDM_KRYLOV_METHOD_BGCRODR:
            infos["krylov_method"] = "bgcrodr";
            break;
            case HPDDM_KRYLOV_METHOD_BFBCG:
            infos["krylov_method"] = "bfbcg";
            break;
            case HPDDM_KRYLOV_METHOD_RICHARDSON:
            infos["krylov_method"] = "richardson";
            break;
            case HPDDM_KRYLOV_METHOD_NONE:
            infos["krylov_method"] = "none";
            break;
        }

        if (infos["Precond"]=="None"){
            infos["GenEO_coarse_size"]="0";
            infos["Coarse_correction"] = "None";
            infos["DDM_local_coarse_size_mean"] = "0";
            infos["DDM_local_coarse_size_max"] = "0";
            infos["DDM_local_coarse_size_min"] = "0";
        }
        else {
            infos["GenEO_coarse_size"]=NbrToStr(size_E);
            int nevi_mean = nevi;
            int nevi_max = nevi;
            int nevi_min = nevi;


            if (rankWorld==0){
                MPI_Reduce(MPI_IN_PLACE, &(nevi_mean),1, MPI_INT, MPI_SUM, 0,this->comm);
                MPI_Reduce(MPI_IN_PLACE, &(nevi_max),1, MPI_INT, MPI_MAX, 0,this->comm);
                MPI_Reduce(MPI_IN_PLACE, &(nevi_min),1, MPI_INT, MPI_MIN, 0,this->comm);
            }
            else{
                MPI_Reduce(&(nevi_mean), &(nevi_mean),1, MPI_INT, MPI_SUM, 0,this->comm);
                MPI_Reduce(&(nevi_max), &(nevi_max),1, MPI_INT, MPI_MAX, 0,this->comm);
                MPI_Reduce(&(nevi_min), &(nevi_min),1, MPI_INT, MPI_MIN, 0,this->comm);
            }

            infos["DDM_local_coarse_size_mean"] = NbrToStr((double)nevi_mean/(double)sizeWorld);
            infos["DDM_local_coarse_size_max"] = NbrToStr(nevi_max);
            infos["DDM_local_coarse_size_min"] = NbrToStr(nevi_min);

            switch (opt.val("schwarz_coarse_correction",-1)) {
                case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                infos["Coarse_correction"] = "Balanced";
                break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                infos["Coarse_correction"] = "Deflated";
                break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                infos["Coarse_correction"] = "Additive";
                break;
                default:
                infos["Coarse_correction"] = "None";
                infos["GenEO_coarse_size"]="0";
                infos["DDM_local_coarse_size_mean"]="0";
                infos["DDM_local_coarse_size_max"]="0";
                infos["DDM_local_coarse_size_min"]="0";
            }
        }
        infos["htool_solver"]="ddm";

    }

  	void print_infos() const{
    	if (hpddm_op.HA.get_rankworld()==0){
    		for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
    			std::cout<<it->first<<"\t"<<it->second<<std::endl;
    		}
        std::cout << std::endl;
    	}
    }

    void save_infos(const std::string& outputname,std::ios_base::openmode mode = std::ios_base::app, const std::string& sep=" = ") const{
    	if (hpddm_op.HA.get_rankworld()==0){
    		std::ofstream outputfile(outputname, mode);
    		if (outputfile){
    			for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
    				outputfile<<it->first<<sep<<it->second<<std::endl;
    			}
    			outputfile.close();
    		}
    		else{
    			std::cout << "Unable to create "<<outputname<<std::endl;
    		}
    	}
    }

    void add_infos(std::string key, std::string value) const{
        if (hpddm_op.HA.get_rankworld()==0){
            if (infos.find(key)==infos.end()){
                infos[key]=value;
            }
            else{
                infos[key]=NbrToStr( StrToNbr<double>(infos[key])+StrToNbr<double>(value));
            }
        }
    }

    void set_infos(std::string key, std::string value) const{
        if (hpddm_op.HA.get_rankworld()==0){
            infos[key]=value;
        }
    }

    std::string get_infos(const std::string& key) const{
        if (hpddm_op.HA.get_rankworld()==0){
            return infos[key];
        }
        return "";
    }

    int get_nevi() const {return nevi;}

};

}
#endif
