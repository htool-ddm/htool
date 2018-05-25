#ifndef HTOOL_WRAPPER_HPDDM_HPP
#define HTOOL_WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'F'
#define HPDDM_DENSE 1
#define HPDDM_FETI 0
#define HPDDM_BDD 0
// #define HPDDM_DENSE 1
#define LAPACKSUB
#define DSUITESPARSE
#define EIGENSOLVER 1
#include <HPDDM.hpp>
#include "../types/hmatrix.hpp"
#include "../types/matrix.hpp"
#include "../solvers/proto_ddm.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class DDM;

template<template<typename> class LowRankMatrix, typename T>
class Proto_DDM;

template< template<typename> class LowRankMatrix, typename T>
class HPDDMDense : public HpDense<T, 'G'> {
private:
    const HMatrix<LowRankMatrix,T>& HA;
    std::vector<T>* in_global,*buffer;


public:
    typedef  HpDense<T, 'G'> super;

    HPDDMDense(const HMatrix<LowRankMatrix,T>& A):HA(A){
        in_global = new std::vector<T> ;
        buffer = new std::vector<T>;
    }
    ~HPDDMDense(){delete in_global;in_global=nullptr;delete buffer;buffer=nullptr;}


    virtual void GMV(const T* const in, T* const out, const int& mu = 1) const override {
        int local_size = HA.get_local_size();

        // Tranpose without overlap
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    (*buffer)[i+j*mu]=in[i*this->getDof()+j];
                }
            }
        }

        // All gather
        if (mu==1){// C'est moche
            HA.mvprod_local(in,out,in_global->data(),mu);
        }
        else{
            HA.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        }



        // Tranpose
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
                }
            }
        }
        bool allocate = this->getMap().size() > 0 && this->getBuffer()[0] == nullptr ? this->setBuffer() : false;
        this->scaledExchange(out, mu);
        if(allocate)
            this->clearBuffer(allocate);
    }

    void exchange(T* const out, const int& mu = 1){
    this->template scaledExchange<true>(out, mu);
    }

    friend class DDM<LowRankMatrix,T>;

};

template< template<typename> class LowRankMatrix, typename T>
class Proto_HPDDM : public HpDense<T, 'G'> {
private:
    const HMatrix<LowRankMatrix,T>& HA;
    std::vector<T>* in_global;
    Proto_DDM<LowRankMatrix,T>& P;
    mutable std::map<std::string, std::string> infos;

public:
    typedef  HpDense<T, 'G'> super;

    Proto_HPDDM(const HMatrix<LowRankMatrix,T>& A,  Proto_DDM<LowRankMatrix,T>& P0):HA(A),P(P0){
        in_global = new std::vector<T> (A.nb_cols());
        P.init_hpddm(*this);
    }
    ~Proto_HPDDM(){delete in_global;}

    void GMV(const T* const in, T* const out, const int& mu = 1) const {
        HA.mvprod_local(in,out,in_global->data(),1);
    }

    template<bool = true>
    void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
        P.apply(in,out);
        // std::copy_n(in, this->_n, out);
    }

    void build_coarse_space(Matrix<T>& mass, Matrix<T>& Bi){
        // Coarse space
        P.build_coarse_space(mass,Bi);
    }

    void facto_one_level(){
        P.facto_one_level();
    }
    void solve(const T* const rhs, T* const x, const int& mu=1 ){
        //
        int rankWorld = HA.get_rankworld();
        int sizeWorld = HA.get_sizeworld();
        int offset  = HA.get_local_offset();
        int nb_cols = HA.nb_cols();
        int nb_rows = HA.nb_rows();
        double time = MPI_Wtime();
        int n = P.get_n();
        int n_inside = P.get_n_inside();
        double time_vec_prod = StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"));
        int nb_vec_prod =  StrToNbr<int>(HA.get_infos("nbr_mat_vec_prod"));
        P.timing_Q=0;
        P.timing_one_level=0;

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n,0);
        std::vector<T> local_rhs(n,0);

        // Permutation
        HA.source_to_cluster_permutation(rhs,rhs_perm.data());
        std::copy_n(rhs_perm.begin()+offset,n_inside,local_rhs.begin());


        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu,HA.get_comm());

        // // Delete the overlap (useful only when mu>1 and n!=n_inside)
        // for (int i=0;i<mu;i++){
        //     std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        // }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int>  displs(sizeWorld);

        displs[0] = 0;

        for (int i=0; i<sizeWorld; i++) {
        recvcounts[i] = (HA.get_MasterOffset_t(i).second);
        if (i > 0)
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        MPI_Allgatherv(x_local.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_global->data() , &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), HA.get_comm());

        // Permutation
        HA.cluster_to_target_permutation(in_global->data(),x);

        // Timing
        HPDDM::Option& opt = *HPDDM::Option::get();
        time = MPI_Wtime()-time;
        infos["Solve"] = NbrToStr(time);
        infos["Nb_it"] = NbrToStr(nb_it);
        infos["nb_mat_vec_prod"] = NbrToStr(StrToNbr<int>(HA.get_infos("nbr_mat_vec_prod"))-nb_vec_prod);
        infos["mean_time_mat_vec_prod"] = NbrToStr((StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"))-time_vec_prod)/(StrToNbr<double>(HA.get_infos("nbr_mat_vec_prod"))-nb_vec_prod));
        switch (opt.val("schwarz_method",0)) {
            case HPDDM_SCHWARZ_METHOD_NONE:
            infos["Precond"] = "none";
            break;
            case HPDDM_SCHWARZ_METHOD_RAS:
            infos["Precond"] = "ras";
            break;
            case HPDDM_SCHWARZ_METHOD_ASM:
            infos["Precond"] = "asm";
            break;
            case HPDDM_SCHWARZ_METHOD_OSM:
            infos["Precond"] = "osm";
            break;
            case HPDDM_SCHWARZ_METHOD_ORAS:
            infos["Precond"] = "oras";
            break;
            case HPDDM_SCHWARZ_METHOD_SORAS:
            infos["Precond"] = "soras";
            break;
        }

        //
        if (infos["Precond"]=="none"){
            infos["GenEO_nu"]="None";
            infos["Coarse_correction"]="None";
        }
        else{
            infos["GenEO_nu"]=NbrToStr(P.get_nevi());
            switch (opt.val("schwarz_coarse_correction",42)) {
                case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                infos["Coarse_correction"] = "Balanced";
                break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                infos["Coarse_correction"] = "Additive";
                break;
                case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                infos["Coarse_correction"] = "Deflated";
                break;
                default:
                infos["Coarse_correction"] = "None";
                infos["GenEO_nu"] = "None";
                break;
            }

        }

        double timing_one_level=P.get_timing_one_level();
        double timing_Q=P.get_timing_Q();
        double maxtiming_one_level,maxtiming_Q;
        // Timing
        MPI_Reduce(&(timing_one_level), &(maxtiming_one_level), 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
        MPI_Reduce(&(timing_Q), &(maxtiming_Q), 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());

        infos["DDM_apply_one_level_max" ]= NbrToStr(maxtiming_one_level);
        infos["DDM_apply_Q_max" ]= NbrToStr(maxtiming_Q);
        infos["DDM_total_time_max"]=NbrToStr(maxtiming_one_level+maxtiming_Q+(StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"))-time_vec_prod));

    }

    void print_infos() const{
        if (HA.get_rankworld()==0){
            for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
                std::cout<<it->first<<"\t"<<it->second<<std::endl;
            }
            for (std::map<std::string,std::string>::const_iterator it = P.get_infos().begin() ; it != P.get_infos().end() ; ++it){
                std::cout<<it->first<<"\t"<<it->second<<std::endl;
            }
        std::cout << std::endl;
        }
    }
    void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::out) const{
    	if (HA.get_rankworld()==0){
    		std::ofstream outputfile(outputname, mode);
    		if (outputfile){
    			for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
    				outputfile<<it->first<<" : "<<it->second<<std::endl;
    			}
                for (std::map<std::string,std::string>::const_iterator it = P.get_infos().begin() ; it != P.get_infos().end() ; ++it){
                    outputfile<<it->first<<" : "<<it->second<<std::endl;
                }
    			outputfile.close();
    		}
    		else{
    			std::cout << "Unable to create "<<outputname<<std::endl;
    		}
    	}
    }


    void add_infos(std::string key, std::string value) const{
        if (HA.get_rankworld()==0){
            if (infos.find(key)==infos.end()){
                infos[key]=value;
            }
            else{
                infos[key]+= value;
            }
        }
    }
};

template< template<typename> class LowRankMatrix, typename T>
class Calderon : public HPDDM::EmptyOperator<T> {
private:
    const HMatrix<LowRankMatrix,T>& HA;
    const HMatrix<LowRankMatrix,T>& HB;
    Matrix<T>& M;
    std::vector<int> _ipiv;
    std::vector<T>* in_global,*buffer;
    mutable std::map<std::string, std::string> infos;

public:

    Calderon(const HMatrix<LowRankMatrix,T>& A,  const HMatrix<LowRankMatrix,T>& B,  Matrix<T>& M0):HPDDM::EmptyOperator<T>(A.get_local_size()),HA(A),HB(B),M(M0),_ipiv(M.nb_rows()){
        in_global = new std::vector<T> ;
        buffer = new std::vector<T>;

        // LU facto
        int size = M.nb_rows();
        int lda=M.nb_rows();
        int info;
        HPDDM::Lapack<Cplx>::getrf(&size,&size,M.data(),&lda,_ipiv.data(),&info);


    }

    ~Calderon(){delete in_global;delete buffer;}


    void GMV(const T* const in, T* const out, const int& mu = 1) const {
        int local_size = HA.get_local_size();

        // Tranpose without overlap
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    (*buffer)[i+j*mu]=in[i*this->getDof()+j];
                }
            }
        }

        // All gather
        if (mu==1){// C'est moche
            HA.mvprod_local(in,out,in_global->data(),mu);
        }
        else{
            HA.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        }



        // Tranpose
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
                }
            }
        }

    }

    template<bool = true>
    void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
        int local_size = HB.get_local_size();
        int offset = HB.get_local_offset();
        // Tranpose
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    (*buffer)[i+j*mu]=in[i*this->getDof()+j];
                }
            }
        }

        // M^-1
        HA.local_to_global(in, in_global->data(),mu);
        const char l='N';
        int n= M.nb_rows();
        int lda=M.nb_rows();
        int ldb=M.nb_rows();
        int nrhs =mu ;
        int info;
        HPDDM::Lapack<T>::getrs(&l,&n,&nrhs,M.data(),&lda,_ipiv.data(),in_global->data(),&ldb,&info);

        // All gather
        if (mu==1){// C'est moche
            HB.mvprod_local(in_global->data()+offset,out,in_global->data()+M.nb_rows(),mu);
        }
        else{
            HB.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        }

        // M^-1
        HA.local_to_global(out, in_global->data(),mu);
        HPDDM::Lapack<T>::getrs(&l,&n,&nrhs,M.data(),&lda,_ipiv.data(),in_global->data()+M.nb_rows(),&ldb,&info);

        // Tranpose
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
                }
            }
        }

    }


    void solve(const T* const rhs, T* const x, const int& mu=1 ){
        //
        int rankWorld = HA.get_rankworld();
        int sizeWorld = HA.get_sizeworld();
        int offset  = HA.get_local_offset();
        int nb_cols = HA.nb_cols();
        int nb_rows = HA.nb_rows();
        int n_local = this->_n;
        double time = MPI_Wtime();
        double time_vec_prod = StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"));
        int nb_vec_prod =  StrToNbr<int>(HA.get_infos("nbr_mat_vec_prod"));
        in_global->resize(nb_cols*2*mu);
        buffer->resize(n_local*(mu==1 ? 1 : 2*mu));

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n_local,0);
        std::vector<T> local_rhs(n_local,0);

        // Permutation
        HA.source_to_cluster_permutation(rhs,rhs_perm.data());
        std::copy_n(rhs_perm.begin()+offset,n_local,local_rhs.begin());

        // TODO avoid com here
        // for (int i=0;i<n-n_inside;i++){
        //   local_rhs[i]=rhs_perm[]
        // }
        // this->exchange(local_rhs.data(), mu);

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu,HA.get_comm());

        // // Delete the overlap (useful only when mu>1 and n!=n_inside)
        // for (int i=0;i<mu;i++){
        //     std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        // }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int>  displs(sizeWorld);

        displs[0] = 0;

        for (int i=0; i<sizeWorld; i++) {
        recvcounts[i] = (HA.get_MasterOffset_t(i).second);
        if (i > 0)
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        MPI_Allgatherv(x_local.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_global->data() , &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), HA.get_comm());

        // Permutation
        HA.cluster_to_target_permutation(in_global->data(),x);

        // Timing
        HPDDM::Option& opt = *HPDDM::Option::get();
        time = MPI_Wtime()-time;
        infos["Solve"] = NbrToStr(time);
        infos["Nb_it"] = NbrToStr(nb_it);
        infos["nb_mat_vec_prod"] = NbrToStr(StrToNbr<int>(HA.get_infos("nbr_mat_vec_prod"))-nb_vec_prod);
        infos["mean_time_mat_vec_prod"] = NbrToStr((StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"))-time_vec_prod)/(StrToNbr<double>(HA.get_infos("nbr_mat_vec_prod"))-nb_vec_prod));


    }

    void print_infos() const{
        if (HA.get_rankworld()==0){
            for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
                std::cout<<it->first<<"\t"<<it->second<<std::endl;
            }
            std::cout << std::endl;
        }
    }
    void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::out) const{
    	if (HA.get_rankworld()==0){
    		std::ofstream outputfile(outputname, mode);
    		if (outputfile){
    			for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
    				outputfile<<it->first<<" : "<<it->second<<std::endl;
    			}
    			outputfile.close();
    		}
    		else{
    			std::cout << "Unable to create "<<outputname<<std::endl;
    		}
    	}
    }
};

}
#endif
