#ifndef HTOOL_LOADING_HPP
#define HTOOL_LOADING_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include "matrix.hpp"
#include "user.hpp"
#include "parametres.hpp"

namespace htool {

template<typename T>
void LoadMatrix(const char* filename, Matrix<T>& m){

	int NbRow, NbCol;
	std::string      line;
	int        j0,k0;
	T        val;

	// Ouverture fichier
	std::ifstream file; file.open(filename);
	if(!file.good()){
		std::cout << "Cannot open file."<<std::endl;
	}

	// Lecture parametres
	int ndofperelt=GetNdofPerElt();

	// Lecture nombre de lignes et de colonnes
	file >> NbRow; file >> NbCol;
	m.resize(NbRow,NbCol);

	getline(file,line);
	getline(file,line);
	while(!file.eof()){

		// Lecture de la ligne
		std::istringstream iss(line);

		// Pour chaque ligne, stockage
		// du bloc d'interaction
		iss >> j0; j0 = ndofperelt*(j0-1);
		iss >> k0; k0 = ndofperelt*(k0-1);

		for(int j=0; j<ndofperelt; j++){
			for(int k=0; k<ndofperelt; k++){
				iss >> val;
				m(j0+j,k0+k) = val;
			}
		}
		getline(file,line);
	}

	file.close();
}

//==================================================//
//
//  DESCRIPTION:
//  Charge la matrice sparse
//
//  INPUT:
//  filename: nom du fichier de la matrice sparse de Ibtihel
//
//  OUTPUT:
//  m: matrice sparse
//
//==================================================//

// void LoadSpMatrix(const char* filename, SpMatrix& m){
//
// 	int NbRow, NbCol;
// 	string      line;
// 	int        j0,k0;
// 	Cplx         val;
//
// 	// Ouverture fichier
// 	ifstream file; file.open(filename);
// 	if(!file.good()){
// 		cout << "LoadSpMatrix in loading.hpp: error opening the matrix file" << endl;
//         cout << filename << endl;
// 		abort();}
//
// 	// Lecture nombre de lignes et de colonnes
// 	file >> NbRow; file >> NbCol;
// 	m.resize(NbRow,NbCol,NbRow*max(NbCol/10,10));
//
// 	// Lecture parametres
// 	int ndofperelt=GetNdofPerElt();
//
// 	int NbCoef=0;
//
// 	getline(file,line);
// 	getline(file,line);
// 	while(!file.eof()){
//
// 		// Lecture de la ligne
// 		istringstream iss(line);
//
// 		// Pour chaque ligne, stockage
// 		// du bloc d'interaction
// 		iss >> j0; j0 = ndofperelt*(j0-1);
// 		iss >> k0; k0 = ndofperelt*(k0-1);
//
// 		for(int j=0; j<ndofperelt; j++){
// 			for(int k=0; k<ndofperelt; k++){
// 				iss >> val;
// 				m.I_(NbCoef) = j0+j;
// 				m.J_(NbCoef) = k0+k;
// 				m.K_(NbCoef) = val;
// 				//m(j0+j,k0+k) = val;
// 				// std::cout <<m.I_(NbCoef)<<" "<<m.J_(NbCoef)<<" "<<m.K_(NbCoef)<< std::endl;
// 				NbCoef++;
// 			}
// 		}
// 		getline(file,line);
//
// 		if (NbCoef+10>NbRow*max(NbCol/10,10)){
// 			m.resize(NbRow,NbCol,std::min(NbCoef*2,NbRow*NbCol));
// 		}
//
// 	}
//
// 	file.close();
//
// 	m.resize(NbRow,NbCol,NbCoef);
// }


//==================================================//
//
//  DESCRIPTION:
//  Charge les donnees geometriques
//  associees au nuage de points
//
//  INPUT:
//  filename: nom du fichier de maillage
//
//  OUTPUT:
//  x: nuage de points (centre des elements)
//  r: rayon de champ proche associe a chaque point
//
//==================================================//

// void LoadPoints(const char* filename, vectR3& x, vectReal& r){
//
// 	x.clear(); r.clear();
// 	int NbElt, NbPt, poubelle;
// 	R3 Pt[4]; R3 Ctr; Real Rmax,Rad;
//
// 	// Ouverture fichier
// 	ifstream file; file.open(filename);
// 	if(!file.good()){
// 		cout << "LoadPoints in loading.hpp: " ;
// 		cout << "error opening the geometry file\n";
// 		abort();}
//
// 	// Nombre d'elements
// 	file >> NbElt;
//
// 	// Lecture elements
// 	for(int e=0; e<NbElt; e++){
// 		Ctr=0.; file >> poubelle;
// 		file >> NbPt;
//
// 		// Calcul centre element
// 		for(int j=0; j<NbPt; j++){
// 			file >> poubelle; file>>Pt[j];
// 			Ctr+= (1./Real(NbPt))*Pt[j];}
//
// 		// Ajout du point
// 		x.push_back(Ctr);
//
// 		// Calcul du rayon champ
// 		// proche associe a l'element
// 		Rmax = norm(Ctr-Pt[0]);
//
// 		for(int j=1; j<NbPt; j++){
// 			Rad = norm(Ctr-Pt[j]);
// 			if(Rad>Rmax){Rmax=Rad;}}
//
// 		r.push_back(Rmax);
//
// 		// Separateur inter-element
// 		if(e<NbElt-1){file >> poubelle;}
// 	}
//
// 	// Fermeture fichier
// 	file.close();
//
// }

}

#endif
