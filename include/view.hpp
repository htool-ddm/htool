#ifndef VIEW_HPP
#define VIEW_HPP

#include <algorithm>

namespace htool {




void LoadMesh(std::string inputname, std::vector<R3>&  X, std::vector<N4>&  Elt, std::vector<int>& NbPt) {
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
}




	
class Palette{
	public:
		unsigned int n;
		std::vector<R3> colors;
		Palette() {}
		Palette(const unsigned int nb, const std::vector<R3>& cols) {
			n = nb;
			colors = cols;	
		}
		
		R3 get_color(float z) const{
			int i=(int)(z*(n-1));
			float t=z*(n-1)-i;
			R3 col = ((1-t)*colors[i]+t*colors[i])/255;
			return col;
		}
};

Palette default_palette;
	
class GLMesh {
	private:
		std::vector<R3>  X;
		std::vector<N4>  Elts;
		std::vector<int> NbPts;
		std::vector<R3>  normals;
		std::vector<int> labels;
		unsigned int nblabels;
		unsigned int visudepth;
		R3 lbox, ubox;
		Palette& palette;
		const Cluster* cluster;
		
	public:
		GLMesh(const GLMesh& m) : palette(m.palette){
			X = m.X;
			Elts = m.Elts;
			NbPts = m.NbPts;
			normals = m.normals;
			labels = m.labels;
			nblabels = m.nblabels;
			visudepth = m.visudepth;
			lbox = m.lbox;
			ubox = m.ubox;
			cluster = m.cluster;			
		}
		
		~GLMesh() {
			X.clear();
			Elts.clear();
			NbPts.clear();
			normals.clear();
			labels.clear();
			if (cluster != NULL) {
				delete cluster;
				cluster = NULL;
			}
		}
	
		GLMesh(std::vector<R3>& X0, std::vector<N4>&  Elts0, std::vector<int>& NbPts0, const Cluster* cluster0 = NULL, Palette& palette0 = default_palette) 
			: palette(palette0), cluster(cluster0){
			X = X0;
			Elts = Elts0;
			NbPts = NbPts0;
			lbox = 1.e+30;
			ubox = -1.e+30;
			for (int i=0; i<X.size(); i++){
				if (X[i][0] < lbox[0]) lbox[0] = X[i][0];
				if (X[i][0] > ubox[0]) ubox[0] = X[i][0];
				if (X[i][1] < lbox[1]) lbox[1] = X[i][1];
				if (X[i][1] > ubox[1]) ubox[1] = X[i][1];
				if (X[i][2] < lbox[2]) lbox[2] = X[i][2];
				if (X[i][2] > ubox[2]) ubox[2] = X[i][2];
			}
			
			normals.resize(Elts.size());
			labels.resize(Elts.size());
			R3 v1,v2;
			for (int i=0; i<Elts.size(); i++){
				v1 = X[Elts[i][1]]-X[Elts[i][0]];
				v2 = X[Elts[i][2]]-X[Elts[i][0]];
				normals[i] = v1^v2;
				normals[i] /= norm(normals[i]);
				labels[i] = 0;
			}
			
			nblabels = 1;
			visudepth = 0;
		}
		
		const R3& get_lbox() const{
			return lbox;
		}
		
		const R3& get_ubox() const{
			return ubox;
		}
		
		const unsigned int& get_visudepth() const{
			return visudepth;
		}
		
		const Cluster* get_cluster() const{
			return cluster;
		}
		
		void set_cluster(const Cluster* c) {
			cluster = c;
		}
		
		void TraversalBuildLabel(const Cluster& t){
			if(depth_(t)<visudepth && !t.IsLeaf()){
				TraversalBuildLabel(son_(t,0));
				TraversalBuildLabel(son_(t,1));
			}
			else {
				for(int i=0; i<num_(t).size()/GetNdofPerElt(); i++)
					labels[num_(t)[GetNdofPerElt()*i]/GetNdofPerElt()] = nblabels;
				nblabels++;
			}
		}

		void set_visudepth(const unsigned int depth){
			if (cluster == NULL)
				std::cerr << "No cluster for this GLMesh" << std::endl;
			else {
				std::cout << "Depth set to " << depth << std::endl;
				visudepth = depth;
				nblabels = 1;
				TraversalBuildLabel(*cluster);
			}
		}
		
		void draw() const{
			for (int i=0;i<Elts.size();i++) {
				if (NbPts[i] == 3)
					glBegin(GL_TRIANGLES);
				else if (NbPts[i] == 4)
					glBegin(GL_QUADS);
				else
					assert(0);
				
				glNormal3d((GLfloat)normals[i][0], (GLfloat)normals[i][1], (GLfloat)normals[i][2]);
				
				R3 col = palette.get_color(1.*labels[i]/nblabels);
				glColor3f(col[0],col[1],col[2]);
					
				for (int j=0;j<NbPts[i];j++){
					glVertex3f(X[Elts[i][j]][0], X[Elts[i][j]][1], X[Elts[i][j]][2]);
				}
				glEnd();
			}
			
			glColor3f(0,0,0);
			for (int i=0;i<Elts.size();i++) {
			    glBegin(GL_LINE_LOOP);
			    for (int j=0;j<NbPts[i];j++)
					glVertex3f(X[Elts[i][j]][0], X[Elts[i][j]][1], X[Elts[i][j]][2]);
    			glEnd();
				
			}
		}

};

class Camera{
	public:
		R3 eye,x,y,z,center,up;
		
		Camera() {
			eye = 0;
			center = 1;
			x = 1;
			y = 0;
			z = 0;	
		};
		
		Camera(const R3& eye0, const R3& center0, const R3& up0){
			eye = eye0;
			center = center0;
			up = up0;
			z = center-eye;
			y = up;
			z /= norm(z);
			y /= norm(y);
			x = y^z;
			x /= norm(x);
		}
		
		void set(const R3& eye0, const R3& center0, const R3& up0){
			eye = eye0;
			center = center0;
			up = up0;
			z = center-eye;
			y = up;
			z /= norm(z);
			y /= norm(y);
			x = y^z;
			x /= norm(x);
		}
		
		void center_on(const GLMesh& mesh){
			center = 0.5*(mesh.get_lbox()+mesh.get_ubox());
			up = 0;
			up[2] = 1;
			eye = 0.2*(mesh.get_lbox()+mesh.get_ubox());
			eye[2] = 0;
			z = center-eye;
			y = up;
			z /= norm(z);
			y /= norm(y);
			x = y^z;
			x /= norm(x);
		}		
};

class Project{
	private:
		GLMesh* mesh;
		std::string name;
		Camera cam;
	public:
		Project(const char* s){
			mesh = NULL;
			name = s;			
		}
		
		~Project(){
			if (mesh != NULL) {
				delete mesh;
				mesh = NULL;
			}
		}
	
		GLMesh* get_mesh() const{
			return mesh;
		}
		
		void set_mesh(const GLMesh& m) {
			if (mesh != NULL)
				delete mesh;
			mesh = new GLMesh(m);
		}		
		
		std::string& get_name() {
			return name;	
		}
		
		Camera& get_camera() {
			return cam;	
		}
		
		void set_camera(const Camera& c){
			cam = c;
		}
		
		void center_view_on_mesh() {
			if (mesh != NULL)
				cam.center_on(*mesh);	
		}
};

class Scene{
	private:
		static std::list<Project> projects;
		static Project* active_project;
		static Real motionx, motiony;
		static GLUI_FileBrowser *fbmesh;
		static GLUI_FileBrowser *fbmat;
		static GLUI_List* list_projects;
		static GLUI_EditText* text_project;
	public:	
		Scene() {}
		
		static void set_active_project(Project* p) {
			active_project = p;	
		}
		
		static void set_mesh(const GLMesh& mesh){
			if (active_project == NULL)
				std::cerr << "No active project" << std::endl;
			else {
				active_project->set_mesh(mesh);
				active_project->center_view_on_mesh();
			}
		}
		
		static void draw_mesh(){
			GLMesh* mesh = NULL;
			if (active_project != NULL)
				mesh = active_project->get_mesh();
			if (mesh != NULL)
				mesh->draw();
		}
		
		static void motion(int x, int y){
			if (active_project == NULL)
				return;
			Camera& cam = active_project->get_camera();
			R3 trans = cam.center;
			cam.center = 0;
			cam.eye = cam.eye - trans;			
			Real scale = norm(cam.eye);
			cam.eye /= scale;
			
			if (x != motionx) {
        		float depl = 0.1;
        		if (x > motionx)
                	depl *= -1;
        		float xx = cam.eye[0]*cos(depl) - cam.eye[1]*sin(depl);
        		float yy = cam.eye[0]*sin(depl) + cam.eye[1]*cos(depl);
        		cam.eye[0] = xx;
        		cam.eye[1] = yy;
				cam.center = 0;
			
				R3 zz = 0;
				zz[2] = cam.y[2];
				zz /= norm(zz);
				cam.z = cam.center-cam.eye;
				cam.z /= norm(cam.z);
				cam.x = zz^cam.z;
				cam.x /= norm(cam.x);
				cam.y = cam.x^cam.z;
				cam.y *= -1;
				cam.y /= norm(cam.y);
				
				cam.center = cam.eye+cam.z;				
				cam.up = cam.y;					
			}
						
			if (y != motiony) {
				float depl = 0.1;
				R3 oldeye = cam.eye;
				if (y > motiony)
					depl *= -1;
				R3 uu = cam.y;
				uu /= -tan(std::abs(depl)/2);
				uu += cam.z;
				uu /= norm(uu);
				float nor = 2*norm(cam.eye)*sin(depl/2);
				uu = nor*uu;
				cam.eye += uu;
				cam.center = 0;
				nor = norm(oldeye);
				cam.eye /= norm(cam.eye);
				cam.eye = nor*cam.eye;
				cam.z = cam.center-cam.eye;
				cam.z /= norm(cam.z);
				cam.y = cam.x^cam.z;
				cam.y *= -1;
				cam.y /= norm(cam.y);
				
				cam.center = cam.eye+cam.z;
				cam.up = cam.y;
			}
			
			cam.eye = scale*cam.eye;
			
			cam.center += trans;
			cam.eye += trans;
			
			motionx = x;
			motiony = y;

		}
		
		static void keyboard(unsigned char key, int , int ){
			GLMesh* mesh = NULL;
			if (active_project != NULL)
				mesh = active_project->get_mesh();
    		switch(key){
    			case 'Q':case'q':
 						
        			break;
    			case 'D':case'd':

        			break;
    			case 'Z':case'z':			
   					if (active_project != NULL){
    					Camera& cam = active_project->get_camera();
    					cam.eye += 10*cam.z;
    				}
        			break;
    			case 'S':case's':
   					if (active_project != NULL){
    					Camera& cam = active_project->get_camera();
    					cam.eye += -10*cam.z;
    				}
        			break;
        		case 'A':case'a':
        			if (mesh != NULL)
        				mesh->set_visudepth(std::max(0,(int)mesh->get_visudepth()-1));
        			break;
         		case 'E':case'e':
         			if (mesh != NULL)
        				mesh->set_visudepth(mesh->get_visudepth()+1);
        			break;
    			default:
        			break;
    		}
		}
				
		static void draw(){		
    		// Black background
    		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);//(NEW) setup our buffers
    
    		glMatrixMode(GL_MODELVIEW); 
			glLoadIdentity();
			
			if (active_project != NULL){
				const Camera& cam = active_project->get_camera();
				gluLookAt(cam.eye[0],cam.eye[1],cam.eye[2], cam.center[0],cam.center[1],cam.center[2], cam.up[0],cam.up[1],cam.up[2]);				
			}
			/*
			glTranslatef(0,0,0);
			glRotatef(angleX,1,0,0);
			glRotatef(angleY,0,1,0);
    		*/
    		glClearColor(1.0f,1.0f,1.0f,1.0f);
    		
    		
    		GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    		GLfloat diffuseMaterial[4] = { 1.0, 0., 0., 1.0 };
    		GLfloat MatAmbient[] = { 0.3, 0., 0., 1.0 };
    		//GLfloat lightPosition[] = {(GLfloat)cam.eye[0],(GLfloat)cam.eye[1],(GLfloat)cam.eye[2], 0};
    		GLfloat lightPosition[] = {10,10,5, 0};
    		GLfloat light_diffuse[] = {1,1,1,1};
    		GLfloat light_specular[] = {1,1,1,1};
    		glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    		glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    		glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    		glEnable(GL_LIGHT0);
    		glEnable(GL_LIGHTING);
    		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseMaterial);
    		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, MatAmbient);
    		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);

    		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 25.0);
    		glShadeModel(GL_SMOOTH);
    		
    		draw_mesh();
						
    		glDisable(GL_LIGHT0);
    		glDisable(GL_LIGHTING);
    		      
    		/*
    		glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
			*/ 
			glPopMatrix();
			glFlush();
	
			glutSwapBuffers();
			glutPostRedisplay();//This function is crucial in animation, it refreshes the screen
		}
		
		static void control_cb(int control) {
			/* attach mesh to current project from file */
			if (control == 1) {
				if (active_project == NULL)
					std::cerr << "No active project" << std::endl;
				else{
					std::string str = fbmesh->get_file();
					str = "../matrices/"+str;
					std::vector<R3>  X;
					std::vector<N4>  Elts;
					std::vector<int> NbPts;
					std::cout << "Loading mesh file " << str << " ..." << std::endl;
					LoadMesh(str.c_str(),X,Elts,NbPts);
					GLMesh m(X,Elts,NbPts);
					set_mesh(m);				
				}
  			}
  			/* change active project */
  			if (control == 3) {
  				if (projects.size() > 0){
  					int i = list_projects->get_current_item();
  					auto it = projects.begin();
  					std::advance(it,i);
  					set_active_project(&(*it));
  				}
  			}
  			/* create project */
  			if (control == 4) {
  				const char* name = text_project->get_text();
  				GLUI_List_Item* item = list_projects->get_item_ptr(name);
  				if (item != NULL)
  					std::cerr << "Project \'" << name << "\' already exists" << std::endl;
  				else{
					list_projects->add_item(0,name);
					projects.push_back(Project(name));
  				}
  			}
  			/* delete project */
  			if (control == 5) {
  				if (projects.size() > 0){
  					int i = list_projects->get_current_item();
  					auto it = projects.begin();
  					std::advance(it,i);  				
  					list_projects->delete_item((*it).get_name().c_str());
  					projects.erase(it);
  					set_active_project(NULL);
  				}
  			}
		}

		static void init(int* argc, char **argv){
			std::vector<R3> colors(20);
			colors[0][0]=255;colors[0][1]=102;colors[0][2]=51;
			colors[1][0]=255;colors[1][1]=153;colors[1][2]=51;
			colors[2][0]=255;colors[2][1]=255;colors[2][2]=102;
			colors[3][0]=153;colors[3][1]=204;colors[3][2]=51;
			colors[4][0]=153;colors[4][1]=255;colors[4][2]=0;
			colors[5][0]=51;colors[5][1]=255;colors[5][2]=0;
			colors[6][0]=51;colors[6][1]=255;colors[6][2]=51;
			colors[7][0]=0;colors[7][1]=255;colors[7][2]=102;
			colors[8][0]=0;colors[8][1]=255;colors[8][2]=204;
			colors[9][0]=51;colors[9][1]=255;colors[9][2]=255;
			colors[10][0]=0;colors[10][1]=153;colors[10][2]=255;
			colors[11][0]=0;colors[11][1]=102;colors[11][2]=255;
			colors[12][0]=0;colors[12][1]=51;colors[12][2]=255;
			colors[13][0]=0;colors[13][1]=0;colors[13][2]=255;
			colors[14][0]=102;colors[14][1]=51;colors[14][2]=204;
			colors[15][0]=153;colors[15][1]=51;colors[15][2]=153;
			colors[16][0]=255;colors[16][1]=51;colors[16][2]=255;
			colors[17][0]=255;colors[17][1]=0;colors[17][2]=204;
			colors[18][0]=255;colors[18][1]=0;colors[18][2]=153;
			colors[19][0]=255;colors[19][1]=0;colors[19][2]=0;
			default_palette.n = 20;
			default_palette.colors = colors;
			
		    glutInit(argc, argv);

		    /*Setting up  The Display
		    /    -RGB color model + Alpha Channel = GLUT_RGBA
		    */
		    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE|GLUT_DEPTH);
		
		    //Configure Window Postion
		    glutInitWindowPosition(50, 25);
		
		    //Configure Window Size
		    glutInitWindowSize(1024,768);
		
		    //Create Window
		    int main_window = glutCreateWindow("Hello OpenGL");
    		    
		    GLfloat WHITE[] = {1, 1, 1};
			GLfloat RED[] = {1, 0, 0};
			GLfloat GREEN[] = {0, 1, 0};
			GLfloat MAGENTA[] = {1, 0, 1};
		    
			Real wdt = 100;
		    glClearColor(0,0,0,0);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(70,1,0.001*wdt/2.,1000*wdt/2.);
			glEnable(GL_DEPTH_TEST);//(NEW) Enable depth testing
			glDisable (GL_BLEND);
			
			glLightfv(GL_LIGHT0, GL_DIFFUSE, WHITE);
			glLightfv(GL_LIGHT0, GL_SPECULAR, WHITE);
		  //glMaterialfv(GL_FRONT, GL_SPECULAR, WHITE);
		  //glMaterialf(GL_FRONT, GL_SHININESS, 30);
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
		  
			glEnable(GL_COLOR_MATERIAL);
			glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
		
		    glutMotionFunc(motion);
			glutDisplayFunc(draw);
    		glutKeyboardFunc(keyboard);	
    		
			//GLUI_Master.set_glutIdleFunc (idle);
			//GLUI_Master.set_glutReshapeFunc(glreshape);
			GLUI* glui_v_subwindow = GLUI_Master.create_glui_subwindow(main_window,GLUI_SUBWINDOW_RIGHT);
			glui_v_subwindow->set_main_gfx_window (main_window);
			
			GLUI_Rollout *op_panel_load = glui_v_subwindow->add_rollout("Load");
			fbmesh = new GLUI_FileBrowser(op_panel_load, "Open mesh file", GLUI_PANEL_EMBOSSED, 0, control_cb);
			fbmesh->fbreaddir("../matrices/");
			fbmesh->set_h(90);
			glui_v_subwindow->add_button_to_panel(op_panel_load,"Load mesh", 1,control_cb);
			fbmat = new GLUI_FileBrowser(op_panel_load, "Open matrix file", GLUI_PANEL_EMBOSSED, 0, control_cb);
			fbmat->fbreaddir("../matrices/");
			fbmat->set_h(90);
			glui_v_subwindow->add_button_to_panel(op_panel_load,"Load matrix", 2,control_cb);
			
			GLUI_Rollout *op_panel_projects = glui_v_subwindow->add_rollout("Projects");
			list_projects = new GLUI_List(op_panel_projects,GLUI_PANEL_EMBOSSED,3,control_cb);
			list_projects->set_h(90);
			list_projects->add_item(1,"my project");
			Project p("my project");
			projects.push_back(p);
			set_active_project(&(projects.front()));
			
			text_project = new GLUI_EditText(op_panel_projects,"");
  			text_project->set_text("new project");
			
			glui_v_subwindow->add_button_to_panel(op_panel_projects,"Create", 4,control_cb);
			glui_v_subwindow->add_button_to_panel(op_panel_projects,"Delete", 5,control_cb);
			
			
			
		}
		
		void run() {
			// Loop require by OpenGL
			glutMainLoop();
		}
		
};

std::list<Project> Scene::projects;
Project* Scene::active_project = NULL;
Real Scene::motionx, Scene::motiony;
GLUI_FileBrowser *Scene::fbmesh;
GLUI_FileBrowser *Scene::fbmat;
GLUI_List* Scene::list_projects;
GLUI_EditText* Scene::text_project;
}

#endif