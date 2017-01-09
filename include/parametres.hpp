#ifndef PARAMETRES_HPP
#define PARAMETRES_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include "matrix.hpp"
#include "point.hpp"
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
	static string datapath;
	static string outputpath;
	static string meshname;
	static string matrixname;
	
	friend void LoadParamIO (string inputname);
	friend string GetMatrixPath();
	friend string GetMeshPath();
	friend string GetOutputPath();
	friend string GetMatrixName();
	friend string GetDataPath();
	
	Parametres_IO (); // Constructeur par defaut
	Parametres_IO (string , string, string , string); // Valeurs données à la main par défaut
};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
string Parametres_IO::datapath;
string Parametres_IO::outputpath;
string Parametres_IO::meshname;
string Parametres_IO::matrixname;

Parametres_IO::Parametres_IO(){
	
}

Parametres_IO::Parametres_IO(string datapath0, string outputpath0, string meshname0, string matrixname0){
	datapath=datapath0;
	outputpath=outputpath0;
	meshname=meshname0;
	matrixname=matrixname0;
}

string GetDataPath(){
	return Parametres_IO::datapath;
}
string GetMatrixPath(){
	return Parametres_IO::datapath+"/"+Parametres_IO::matrixname;
}
string GetMatrixName(){
	return Parametres_IO::matrixname;
}
string GetMeshName(){
	return Parametres_IO::meshname;
}
string GetMeshPath(){
	return Parametres_IO::datapath+"/"+Parametres_IO::meshname;
}
string GetOutputPath(){
	return Parametres_IO::outputpath;
}


void LoadParamIO(string inputname){
	ifstream data(inputname.c_str());
	
	// Si le fichier n'existe pas
	if (!data){
		cerr << "Input file doesn't exist" << endl;
		exit(1);
	}
	// Lecture du fichier
	else {
		while (data){
			string strInput;
			getline(data,strInput);
			
			vector<string> line = split (strInput,' ');
			if (!line.empty()){
				if (line.at(0)=="Data_path"){
					Parametres_IO::datapath=line.back();
				}
				else if (line.at(0)=="Output_path"){
					Parametres_IO::outputpath=line.back();
				}
				else if (line.at(0)=="Mesh_name"){
					Parametres_IO::meshname=line.back();
					ifstream Meshname((Parametres_IO::datapath+"/"+Parametres_IO::meshname).c_str());
					if (!Meshname){
						cerr << "Mesh file does not exist" << endl;
						exit(1);
					}
				}
				else if (line.at(0)=="Matrix_name"){
					Parametres_IO::matrixname=line.back();
					ifstream Matrixname((Parametres_IO::datapath+"/"+Parametres_IO::matrixname).c_str());
					if (!Matrixname){
						cerr << "Matrix file does not exist" << endl;
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
	static Real eta;
	static Real epsilon;
	static int maxblocksize;
 	static int minclustersize;	
	
	Parametres();
	Parametres(int, Real, Real, int, int);
	
	friend void LoadParam(string inputname);
	friend Real GetEta();
	friend Real GetEpsilon();
	friend int GetNdofPerElt();
	friend void SetEta(Real);
	friend void SetEpsilon(Real);
	friend void SetNdofPerElt(int);
	friend int GetMaxBlockSize();
	friend void SetMaxBlockSize(int);
	friend int GetMinClusterSize();
	friend void SetMinClusterSize(int);	

};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
Real Parametres::eta;
Real Parametres::epsilon;
int Parametres::ndofperelt;
int Parametres::maxblocksize;
int Parametres::minclustersize;

Parametres::Parametres(){
	
}

Parametres::Parametres(int ndofperelt0, Real eta0, Real epsilon0, int maxblocksize0, int minclustersize0){
	ndofperelt=ndofperelt0;
	eta=eta0;
	epsilon=epsilon0;
	maxblocksize=maxblocksize0;
	minclustersize=minclustersize0;
}


void SetEta(Real eta0){
	Parametres::eta=eta0;
}
void SetEpsilon(Real epsilon0){
	Parametres::epsilon=epsilon0;
}
void SetNdofPerElt(int ndofperelt0){
	Parametres::ndofperelt=ndofperelt0;
}
Real GetEta(){
	return Parametres::eta;
}
Real GetEpsilon(){
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

void LoadParam(string inputname){
	ifstream data(inputname.c_str());
	
	// Si le fichier n'existe pas
	if (!data){
		cerr << "Input file doesn't exist" << endl;
		exit(1);
	}
	// Lecture du fichier
	else {
		while (data){
			string strInput;
			getline(data,strInput);
			
			vector<string> line = split (strInput,' ');
			if (!line.empty()){
				if (line.at(0)=="Epsilon"){
					Parametres::epsilon=StrToNbr<Real>(line.back());
				}
				else if (line.at(0)=="Eta"){
					Parametres::eta=StrToNbr<Real>(line.back());
				}
			}
		}
	}
}

Parametres Parametres_defauts(3,0.5,0.5,100000,3);
}
#endif
