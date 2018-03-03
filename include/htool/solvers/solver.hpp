#ifndef HTOOL_SOLVER_HPP
#define HTOOL_SOLVER_HPP

#include "../types/matrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../wrappers/wrapper_hpddm.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class ISolver{
private:
    ISolver(const ISolver&) = default; // copy constructor
    ISolver& operator=(const ISolver&) = default; // copy assignement operator



protected:
    // Data members
    int n;
    HPDDMDense<LowRankMatrix,T> hpddm_op;
    const MPI_Comm& comm;
    mutable std::map<std::string, std::string> infos;

    // Constructors
    ISolver() = delete;
    ISolver(const HMatrix<LowRankMatrix,T>& hmat_0):n(hmat_0.get_local_size()),hpddm_op(hmat_0),comm(hmat_0.get_comm()){}




public:
    ISolver(ISolver&&) = default; // move constructor
    ISolver& operator=(ISolver&&) = default; // move assignement operator


    // Infos
    void print_infos() const{
        if (hpddm_op.HA.get_rankworld()==0){
            for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
                std::cout<<it->first<<"\t"<<it->second<<std::endl;
            }
        std::cout << std::endl;
        }
    }

    void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::out) const{
        if (hpddm_op.HA.get_rankworld()==0){
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




} // namespace

#endif
