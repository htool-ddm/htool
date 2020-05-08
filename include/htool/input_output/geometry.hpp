#ifndef HTOOL_GEOMETRY_HPP
#define HTOOL_GEOMETRY_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include "../misc/parametres.hpp"
#include "../types/point.hpp"

namespace htool {


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

void LoadPoints(const std::string& filename, std::vector<R3>& x, std::vector<double>& r){

	x.clear(); r.clear();
	int NbElt, NbPt, poubelle;
	R3 Pt[4]; R3 Ctr; double Rmax,Rad;

	// Ouverture fichier
	std::ifstream file; file.open(filename);
	if(!file.good()){
		std::cout << "LoadPoints in loading.hpp: " ;
		std::cout << "error opening the geometry file\n";
		abort();}

	// Nombre d'elements
	file >> NbElt;

	// Lecture elements
	for(int e=0; e<NbElt; e++){
		Ctr.fill(0.); file >> poubelle;
		file >> NbPt;

		// Calcul centre element
		for(int j=0; j<NbPt; j++){
			file >> poubelle; file>>Pt[j];
			Ctr+= (1./double(NbPt))*Pt[j];}

		// Ajout du point
		x.push_back(Ctr);

		// Calcul du rayon champ
		// proche associe a l'element
		Rmax = norm2(Ctr-Pt[0]);

		for(int j=1; j<NbPt; j++){
			Rad = norm2(Ctr-Pt[j]);
			if(Rad>Rmax){Rmax=Rad;}}

		r.push_back(Rmax);

		// Separateur inter-element
		if(e<NbElt-1){file >> poubelle;}
	}

	// Fermeture fichier
	file.close();

}


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

int Load_GMSH_nodes(std::vector<R3>& x, const std::string& filename){

	int size = 0;

  std::istringstream iss;
	std::ifstream file;
	std::string line;

	// Open file
	file.open(filename);
	if(!file.good()){
		std::cout << "Cannot open mesh file\n";
		return 1;
	}

	// Number of elements
	while( line != "$Nodes" ){
    getline(file,line);
	}
	file >> size;
	getline(file,line);
	x.resize(size);

	// Read point
	R3 coord;int dummy;
	getline(file,line);
	for (int p=0;p<size;p++){
		iss.str(line);
		iss >> dummy;
		iss>>coord;
		x[p]=coord;
		iss.clear();
		getline(file,line);
	}

	// Fermeture fichier
	file.close();

	return 0;
}

}

#endif
