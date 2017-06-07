#ifndef PARAMETRES_HPP
#define PARAMETRES_HPP

#include <cassert>
#include "user.hpp"


namespace htool {
//==================================================//
//
//  DESCRIPTION:
//
//
//
//  INPUT:
//  filename: nom du fichier d'input
//
//  OUTPUT:
//
//
//
//==================================================//

class Parametres_IO{
public:
	static std::string datapath;
	static std::string outputpath;
	static std::string meshname;
	static std::string matrixname;

	friend void LoadParamIO (std::string inputname);
	friend std::string GetMatrixPath();
	friend std::string GetMeshPath();
	friend std::string GetOutputPath();
	friend std::string GetMatrixName();
	friend std::string GetDataPath();

	Parametres_IO (); // Constructeur par defaut
	Parametres_IO (std::string , std::string, std::string , std::string); // Valeurs données à la main par défaut
};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
std::string Parametres_IO::datapath;
std::string Parametres_IO::outputpath;
std::string Parametres_IO::meshname;
std::string Parametres_IO::matrixname;

Parametres_IO::Parametres_IO(){

}

Parametres_IO::Parametres_IO(std::string datapath0, std::string outputpath0, std::string meshname0, std::string matrixname0){
	datapath=datapath0;
	outputpath=outputpath0;
	meshname=meshname0;
	matrixname=matrixname0;
}

std::string GetDataPath(){
	return Parametres_IO::datapath;
}
std::string GetMatrixPath(){
	return Parametres_IO::datapath+"/"+Parametres_IO::matrixname;
}
std::string GetMatrixName(){
	return Parametres_IO::matrixname;
}
std::string GetMeshName(){
	return Parametres_IO::meshname;
}
std::string GetMeshPath(){
	return Parametres_IO::datapath+"/"+Parametres_IO::meshname;
}
std::string GetOutputPath(){
	return Parametres_IO::outputpath;
}


void LoadParamIO(std::string inputname){
	std::ifstream data(inputname.c_str());

	// Si le fichier n'existe pas
	if (!data){
		std::cerr << "Input file doesn't exist" << std::endl;
		exit(1);
	}
	// Lecture du fichier
	else {
		while (data){
			std::string strInput;
			getline(data,strInput);

			std::vector<std::string> line = split (strInput,' ');
			if (!line.empty()){
				if (line.at(0)=="Data_path"){
					Parametres_IO::datapath=line.back();
				}
				else if (line.at(0)=="Output_path"){
					Parametres_IO::outputpath=line.back();
				}
				else if (line.at(0)=="Mesh_name"){
					Parametres_IO::meshname=line.back();
					std::ifstream Meshname((Parametres_IO::datapath+"/"+Parametres_IO::meshname).c_str());
					if (!Meshname){
						std::cerr << "Mesh file does not exist" << std::endl;
						exit(1);
					}
				}
				else if (line.at(0)=="Matrix_name"){
					Parametres_IO::matrixname=line.back();
					std::ifstream Matrixname((Parametres_IO::datapath+"/"+Parametres_IO::matrixname).c_str());
					if (!Matrixname){
						std::cerr << "Matrix file does not exist" << std::endl;
						exit(1);
					}
				}
			}
		}
	}
}

Parametres_IO Parametre_IO_defauts("data","output","none","none");

//==================================================//
//
//  DESCRIPTION:
//
//
//
//  INPUT:
//  filename: nom du fichier d'input
//
//  OUTPUT:
//
//
//
//==================================================//

class Parametres{
public:
	static int  ndofperelt;
	static double eta;
	static double epsilon;
	static int maxblocksize;
 	static int minclustersize;

	Parametres();
	Parametres(int, double, double, int, int);

	friend void LoadParam(std::string inputname);
	friend double GetEta();
	friend double GetEpsilon();
	friend int GetNdofPerElt();
	friend void SetEta(double);
	friend void SetEpsilon(double);
	friend void SetNdofPerElt(int);
	friend int GetMaxBlockSize();
	friend void SetMaxBlockSize(int);
	friend int GetMinClusterSize();
	friend void SetMinClusterSize(int);

};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
double Parametres::eta;
double Parametres::epsilon;
int Parametres::ndofperelt;
int Parametres::maxblocksize;
int Parametres::minclustersize;

Parametres::Parametres(){

}

Parametres::Parametres(int ndofperelt0, double eta0, double epsilon0, int maxblocksize0, int minclustersize0){
	ndofperelt=ndofperelt0;
	eta=eta0;
	epsilon=epsilon0;
	maxblocksize=maxblocksize0;
	minclustersize=minclustersize0;
}


void SetEta(double eta0){
	Parametres::eta=eta0;
}
void SetEpsilon(double epsilon0){
	Parametres::epsilon=epsilon0;
}
void SetNdofPerElt(int ndofperelt0){
	Parametres::ndofperelt=ndofperelt0;
}
double GetEta(){
	return Parametres::eta;
}
double GetEpsilon(){
	return Parametres::epsilon;
}
int GetNdofPerElt(){
	return Parametres::ndofperelt;
}

int GetMaxBlockSize(){
	return Parametres::maxblocksize;
}

void SetMaxBlockSize(int maxblocksize0){
	Parametres::maxblocksize=maxblocksize0;
}

int GetMinClusterSize(){
	return Parametres::minclustersize;
}

void SetMinClusterSize(int minclustersize0){
	Parametres::minclustersize=minclustersize0;
}

void LoadParam(std::string inputname){
	std::ifstream data(inputname.c_str());

	// Si le fichier n'existe pas
	if (!data){
		std::cerr << "Input file doesn't exist" << std::endl;
		exit(1);
	}
	// Lecture du fichier
	else {
		while (data){
			std::string strInput;
			getline(data,strInput);

			std::vector<std::string> line = split (strInput,' ');
			if (!line.empty()){
				if (line.at(0)=="Epsilon"){
					Parametres::epsilon=StrToNbr<double>(line.back());
				}
				else if (line.at(0)=="Eta"){
					Parametres::eta=StrToNbr<double>(line.back());
				}
				else if (line.at(0)=="MinClusterSize"){
					Parametres::minclustersize=StrToNbr<double>(line.back());
				}
			}
		}
	}
}

Parametres Parametres_defauts(3,0.5,0.5,10000000,3);
}
#endif
