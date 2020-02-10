#ifndef HTOOL_PARAMETRES_HPP
#define HTOOL_PARAMETRES_HPP

#include <cassert>
#include "user.hpp"


namespace htool {

class Parametres{
public:
	static int  ndofperelt;
	static double eta;
	static double epsilon;
	static int maxblocksize;
 	static int minclustersize;
	static int mintargetdepth; 
	static int minsourcedepth; 

	Parametres();
	Parametres(int, double, double, int, int, int, int);

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
	friend int GetMinTargetDepth();
	friend void SetMinTargetDepth(int);
	friend int GetMinSourceDepth();
	friend void SetMinSourceDepth(int);

};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
double Parametres::eta;
double Parametres::epsilon;
int Parametres::ndofperelt;
int Parametres::maxblocksize;
int Parametres::minclustersize;
int Parametres::mintargetdepth;
int Parametres::minsourcedepth;

Parametres::Parametres(){

}

Parametres::Parametres(int ndofperelt0, double eta0, double epsilon0, int maxblocksize0, int minclustersize0,int mintargetdepth0,int minsourcedepth0){
	ndofperelt=ndofperelt0;
	eta=eta0;
	epsilon=epsilon0;
	maxblocksize=maxblocksize0;
	minclustersize=minclustersize0;
	mintargetdepth=mintargetdepth0;
	minsourcedepth=minsourcedepth0;
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

int GetMinTargetDepth(){
	return Parametres::mintargetdepth;
}

void SetMinTargetDepth(int mintargetdepth0){
	Parametres::mintargetdepth=mintargetdepth0;
}

int GetMinSourceDepth(){
	return Parametres::minsourcedepth;
}

void SetMinSourceDepth(int minsourcedepth0){
	Parametres::minsourcedepth=minsourcedepth0;
}

Parametres Parametres_defauts(1,10,1e-3,1000000,10,0,0);
}
#endif
