#ifndef HTOOL_EXPORT_HPP
#define HTOOL_EXPORT_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include "matrix.hpp"
#include "parametres.hpp"

namespace htool {
//==================================================//
//
//  DESCRIPTION:
//  Convertit le maillage au format de medit
//  pour visualisation. Le fichier de sortie
//  s'appelle "visu.mesh"
//
//  INPUT:
//  filename: nom du fichier de maillage
//
//  OUTPUT:
//  none
//
//  REMARQUE:
//  Cette routine a ete ecrite en considerant
//  que les elements etaient des quadrangles
//
//==================================================//

void ExportMEDIT(std::string inputname, std::string outputname){

	std::vector<R3>  X;
	std::vector<N4>  Elt;
	std::vector<int> NbPt;
	int   num,NbElt,poubelle, NbTri, NbQuad;
	R3    Pt;

	// Ouverture fichier
	std::ifstream infile;
	infile.open(inputname.c_str());
	if(!infile.good()){
		std::cout << "LoadPoints in loading.hpp: error opening the geometry file" << std::endl;
		abort();}

	// Nombre d'elements
	infile >> NbElt;
	Elt.resize(NbElt);
	NbPt.resize(NbElt);

	num=0; NbTri=0; NbQuad=0;
	// Lecture elements
	for(int e=0; e<NbElt; e++){
		infile >> poubelle;
		infile >> NbPt[e];

		if(NbPt[e]==3){NbTri++;}
		if(NbPt[e]==4){NbQuad++;}

		// Calcul centre element
		for(int j=0; j<NbPt[e]; j++){
			infile >> poubelle;
			infile >> Pt;
			Elt[e][j] = num;
			X.push_back(Pt);
			num++;
		}

		// Separateur inter-element
		if(e<NbElt-1){infile >> poubelle;}

	}
	infile.close();

	// Ecriture fichier de sortie
	std::ofstream outfile;
	outfile.open((GetOutputPath()+"/"+outputname).c_str());
	outfile << "MeshVersionFormatted 1\n";
	outfile << "Dimension 3\n";
	outfile << "Vertices\n";
	outfile << X.size() << std::endl;
	for(int j=0; j<X.size(); j++){
		outfile << X[j] << "\t" << 0 << "\n";}

	if(NbQuad>0){
		outfile << std::endl;
		outfile << "Quadrilaterals\n";
		outfile << NbQuad << std::endl;
		for(int j=0; j<NbElt; j++){
			if(NbPt[j]==4){
				for(int k=0; k<4; k++){
					outfile << Elt[j][k]+1 << "\t";
				}
				outfile << 0 << std::endl;
			}
		}
	}

	if(NbTri>0){
		outfile << std::endl;
		outfile << "Triangles\n";
		outfile << NbTri << std::endl;
		for(int j=0; j<NbElt; j++){
			if(NbPt[j]==3){
				for(int k=0; k<3; k++){
					outfile << Elt[j][k]+1 << "\t";}
				outfile << 0 << std::endl;
			}
		}
	}

	outfile.close();

}




//==================================================//
//
//  DESCRIPTION:
//  Convertit le maillage au format gmsh
//  pour visualisation. Le fichier de sortie
//  s'appelle "visu.msh"
//
//  INPUT:
//  filename: nom du fichier de maillage
//
//  OUTPUT:
//  none
//
//  REMARQUE:
//  Cette routine a ete ecrite en considerant
//  que les elements etaient des quadrangles
//
//==================================================//

void ExportGMSH(std::string inputname, std::string outputname){

	std::vector<R3>  X;
	std::vector<N4>  Elt;
	std::vector<int> NbPt;
	int   num,NbElt,poubelle, NbTri, NbQuad;
	R3    Pt;

	// Ouverture fichier
	std::ifstream infile;
	infile.open(inputname.c_str());
	if(!infile.good()){
		std::cout << "LoadPoints in loading.hpp: error opening the geometry file" << std::endl;
		abort();}

	// Nombre d'elements
	infile >> NbElt;
	Elt.resize(NbElt);
	NbPt.resize(NbElt);

	num=0; NbTri=0; NbQuad=0;
	// Lecture elements
	for(int e=0; e<NbElt; e++){
		infile >> poubelle;
		infile >> NbPt[e];

		if(NbPt[e]==3){NbTri++;}
		if(NbPt[e]==4){NbQuad++;}

		// Calcul centre element
		for(int j=0; j<NbPt[e]; j++){
			infile >> poubelle;
			infile >> Pt;
			Elt[e][j] = num;
			X.push_back(Pt);
			num++;
		}

		// Separateur inter-element
		if(e<NbElt-1){infile >> poubelle;}

	}
	infile.close();


	// Ecriture fichier de sortie
	std::ofstream outfile;
	outfile.open((GetOutputPath()+"/"+outputname).c_str());
	outfile << "$MeshFormat\n";
	outfile << "2.2 0 8\n";
	outfile << "$EndMeshFormat\n";
	outfile << "$Nodes\n";
	outfile << X.size() << std::endl;
	for(int j=0; j<X.size(); j++){
		outfile << j+1 << "\t" << X[j] << "\n";}
	outfile << "$EndNodes\n";
	outfile << "$Elements\n";
	outfile << NbElt << std::endl;
	for(int j=0; j<NbElt; j++){
		outfile << j  << "\t";
		if(NbPt[j]==3){outfile << 2  << "\t";}
		if(NbPt[j]==4){outfile << 3  << "\t";}
		outfile << 2  << "\t";
		outfile << 99 << "\t";
		outfile << 1  << "\t";
		for(int k=0; k<NbPt[j]; k++){
			outfile << Elt[j][k]+1 << "\t";}
		outfile << "\n";
	}
	outfile << "$EndElements\n";

}
}
#endif
