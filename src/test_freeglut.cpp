//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>

//#define GLEW_STATIC
//#include <GL/glew.h>

#include <nanogui/nanogui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>

#include "mpi.h"

typedef std::complex<double> K;

#include <htool/htool.hpp>
#include <htool/view.hpp>

using namespace std;
using namespace htool;
using namespace nanogui;

void attach_ui(Scene& s) {
	statics& gv = Scene::gv;
	
	nanogui::Window *w = new nanogui::Window(Scene::gv.screen, "IFP");
	w->setPosition(Eigen::Vector2i(650, 200));
	w->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
			nanogui::Alignment::Middle, 10, 10));
	nanogui::Widget* tools = new nanogui::Widget(w);
	tools->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
                                      nanogui::Alignment::Middle, 0, 6));
   nanogui::Button* b = new nanogui::Button(tools, "Load mesh");
   b->setCallback([&] {
    			if (gv.active_project == NULL)
		std::cerr << "No active project" << std::endl;
			else{
				std::string str = nanogui::file_dialog(
                   {{"txt", "Mesh file"}}, false);
				std::vector<R3>  X;
				std::vector<N4>  Elts;
				std::vector<int> NbPts;
				std::vector<R3> Normals;
				std::vector<R3>  Ctrs;
				std::vector<double> Rays;
				std::cout << "Loading mesh file " << str << " ..." << std::endl;
				LoadMesh(str.c_str(),X,Elts,NbPts,Normals,Ctrs,Rays);
				GLMesh m(X,Elts,NbPts,Normals);
				s.set_mesh(m);
				gv.active_project->set_ctrs(Ctrs);
				gv.active_project->set_rays(Rays);
			}
   });
   
   b = new nanogui::Button(tools, "Load matrix");
   b->setCallback([&] {
    		if (gv.active_project == NULL)
				std::cerr << "No active project" << std::endl;
			else{				
				std::string strmat = nanogui::file_dialog(
                   {{"bin", "Matrix binary file"}}, false);
                   std::cout << "Loading matrix file " << strmat << " ..." << std::endl;              
   				Matrix<K> *A = new Matrix<K>;
                A->bytes_to_matrix(strmat);
                gv.active_project->set_matrix(A);
			}
   });

	gv.screen->performLayout();

}

int main(int argc, char **argv) {
	
	MPI_Init(&argc, &argv);
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    
    //string str = "../matrices/maillage3600FracsV1DN1.txt";
    //string strm = "../matrices/matrice3600FracsV1DN1.bin";
 
 	/*
    string str = "../matrices/maillage450Fracs.txt";
    string strm = "../matrices/matrice450Fracs.bin"; 
    
    bytes_to_matrix(strm,A);
    LoadPoints(str.c_str(),x,r);
    vectInt tab(nb_rows(A));
    for (int j=0;j<x.size();j++){
    	tab[3*j]  = j;
        tab[3*j+1]= j;
        tab[3*j+2]= j;
    }   

	Cluster t(x,r,tab);
	
	LoadMesh(str,X,Elts,NbPts);
	GLMesh m(X,Elts,NbPts);
	m.set_cluster(&t);
	*/
    
    Scene s;
	
	//s.add_mesh(m);
	
	s.init();
	
	attach_ui(s);
	
	s.run();
    
    MPI_Finalize();
    return 0;
}
