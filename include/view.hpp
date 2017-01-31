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
	
	std::cout << NbTri << " triangle(s) and " << NbQuad << " quad(s)" << std::endl;
	
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
		GLuint &VBO, &VAO, &EBO;
		
	public:
		GLMesh(const GLMesh& m) : VBO(m.VBO), VAO(m.VAO), EBO(m.EBO), palette(m.palette){
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
	
		GLMesh(GLuint& VBO0, GLuint& VAO0, GLuint& EBO0, std::vector<R3>& X0, std::vector<N4>&  Elts0, std::vector<int>& NbPts0, const Cluster* cluster0 = NULL, Palette& palette0 = default_palette) 
			: VBO(VBO0), VAO(VAO0), EBO(EBO0), palette(palette0), cluster(cluster0){
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
				set_buffers();
			}
		}
		
		void set_buffers() {		
			int np = NbPts[0];
					
			GLfloat vertices[9*X.size()];
			for (int i=0; i<9*X.size(); i++)
				vertices[i] = 0;

			for (int i=0; i<X.size(); i++)
			for (int j=0; j<3; j++)
				vertices[9*i+j] = X[i][j];
				
			// Normals
			for (int i=0; i<Elts.size(); i++) {
				R3 col = palette.get_color(1.*labels[i]/nblabels);
				for (int j=0; j<NbPts[i]; j++) {
					for (int k=0; k<3; k++)
						vertices[9*Elts[i][j]+3+k] += normals[i][k];
					vertices[9*Elts[i][j]+6] = col[0];
					vertices[9*Elts[i][j]+7] = col[1];
					vertices[9*Elts[i][j]+8] = col[2];
				}
			}

			int sz = (np == 3 ? np : 6);
			GLuint indices[sz*Elts.size()];
					
			if (np == 3) {
				for (int i=0; i<Elts.size(); i++)
				for (int j = 0; j<3; j++)				
					indices[3*i+j] = Elts[i][j];
			}
			else {
				for (int i=0; i<Elts.size(); i++) {
					indices[6*i] = Elts[i][0];
					indices[6*i+1] = Elts[i][1];
					indices[6*i+2] = Elts[i][2];
					indices[6*i+3] = Elts[i][0];
					indices[6*i+4] = Elts[i][3];
					indices[6*i+5] = Elts[i][2];
				}
			}
			
			glGenVertexArrays(1, &VAO);
    		glGenBuffers(1, &VBO);
    		glGenBuffers(1, &EBO);			
    		
    		glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   			
   			glBindVertexArray(VAO);
   			
    		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)0);
    		glEnableVertexAttribArray(0);
    		// Normal attribute
    		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    		glEnableVertexAttribArray(1);
    		
    		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
    		glEnableVertexAttribArray(2);

    		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)0);
    		glEnableVertexAttribArray(0);

    		glBindBuffer(GL_ARRAY_BUFFER, 0); 

    		glBindVertexArray(0);
		}
		
		void draw() const{		
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			if (NbPts[0] == 3)
				glDrawElements(GL_TRIANGLES, 3*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
			else
				glDrawElements(GL_TRIANGLES, 6*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);			
		}

};

class Camera{
	public:
		R3 eye,x,y,z,center,up;
		
		Camera() {
			eye = 0;
			center = 1;
			up = 1;
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
		static nanogui::Screen* screen;
		static GLFWwindow* glwindow;
		static GLint shaderProgram, lightshaderProgram;
		static GLuint VBO, VAO, EBO;
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
				active_project->get_mesh()->set_buffers();
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
				
		static void draw(){
		glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        if (active_project != NULL){
			Camera& cam = active_project->get_camera();

			//glm::vec3 lightPos(cam.center[0],cam.center[1],cam.center[2]+100);
			R3 lp = cam.eye;
			glm::vec3 lightPos(lp[0],lp[1],lp[2]);

        	glUseProgram(shaderProgram);
        
       		//GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
        	GLint lightColorLoc  = glGetUniformLocation(shaderProgram, "lightColor");
        	GLint lightPosLoc    = glGetUniformLocation(shaderProgram, "lightPos");
        	GLint viewPosLoc     = glGetUniformLocation(shaderProgram, "viewPos"); 
        	//glUniform3f(objectColorLoc, 1.0f, 0.5f, 0.31f);
        	glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f);
        	glUniform3f(lightPosLoc,    lightPos.x, lightPos.y, lightPos.z);
        	glUniform3f(viewPosLoc,     cam.eye[0],cam.eye[1],cam.eye[2]);  

			glm::mat4 view = glm::lookAt(glm::vec3(cam.eye[0],cam.eye[1],cam.eye[2]), glm::vec3(cam.center[0],cam.center[1],cam.center[2]), glm::vec3(cam.up[0],cam.up[1],cam.up[2]));
			float wdt = 100;
			//glm::mat4 projection = glm::perspective(0.f, 1.f, 0.1f, 1000.0f);
			//glm::mat4 projection = glm::perspective(70.f, 1.f, 0.001f, 100.0f);
			glm::mat4 projection = glm::perspective(70.f,1.f,(float)(0.001*wdt/2.),(float)(1000*wdt/2.));
			//70,1,0.001*wdt/2.,1000*wdt/2.
			GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        	GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        	GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
	
			glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

			glBindVertexArray(VAO);
			
			glm::mat4 model;
			//model = glm::translate(model, glm::vec3(0.f, 0.f, 0.f));
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
			
			
			GLMesh* mesh = active_project->get_mesh();
			if (mesh != NULL)
				mesh->draw();
			
        	glBindVertexArray(0);
        	
        	
        	/*
        	
        	GLfloat vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
    };
        	
        	GLuint lightVAO;
        	glGenVertexArrays(1, &lightVAO);
    glGenBuffers(1, &VBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    
        	
        	glUseProgram(lightshaderProgram);
        	    
    glGenVertexArrays(1, &lightVAO);
    glBindVertexArray(lightVAO);
    // We only need to bind to the VBO (to link it with glVertexAttribPointer), no need to fill it; the VBO's data already contains all we need.
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Set the vertex attributes (only position data for the lamp))
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0); // Note that we skip over the normal vectors
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    
    modelLoc = glGetUniformLocation(lightshaderProgram, "model");
        viewLoc  = glGetUniformLocation(lightshaderProgram, "view");
        projLoc  = glGetUniformLocation(lightshaderProgram, "projection");
        // Set matrices
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        model = glm::mat4();
        model = glm::translate(model, lightPos);
        model = glm::scale(model, glm::vec3(1.f)); // Make it a smaller cube
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        // Draw the light object (using light's vertex attributes)
        glBindVertexArray(lightVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        */
        	
        	
        	
		
		}	
    		
		}
		
//		static void control_cb(int control) {
//			/* attach mesh to current project from file */
//			if (control == 1) {
//				if (active_project == NULL)
//					std::cerr << "No active project" << std::endl;
//				else{
//					std::string str = fbmesh->get_file();
//					str = "../matrices/"+str;
//					std::vector<R3>  X;
//					std::vector<N4>  Elts;
//					std::vector<int> NbPts;
//					std::cout << "Loading mesh file " << str << " ..." << std::endl;
//					LoadMesh(str.c_str(),X,Elts,NbPts);
//					GLMesh m(X,Elts,NbPts);
//					set_mesh(m);				
//				}
//  			}
//  			/* change active project */
//  			if (control == 3) {
//  				if (projects.size() > 0){
//  					int i = list_projects->get_current_item();
//  					auto it = projects.begin();
//  					std::advance(it,i);
//  					set_active_project(&(*it));
//  				}
//  			}
//  			/* create project */
//  			if (control == 4) {
//  				const char* name = text_project->get_text();
//  				GLUI_List_Item* item = list_projects->get_item_ptr(name);
//  				if (item != NULL)
//  					std::cerr << "Project \'" << name << "\' already exists" << std::endl;
//  				else{
//					list_projects->add_item(0,name);
//					projects.push_back(Project(name));
//  				}
//  			}
//  			/* delete project */
//  			if (control == 5) {
//  				if (projects.size() > 0){
//  					int i = list_projects->get_current_item();
//  					auto it = projects.begin();
//  					std::advance(it,i);  				
//  					list_projects->delete_item((*it).get_name().c_str());
//  					projects.erase(it);
//  					set_active_project(NULL);
//  				}
//  			}
//		}

		void init(int* argc, char **argv){
		
			glfwInit();
    		glfwSetTime(0);
    		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    		glfwWindowHint(GLFW_SAMPLES, 4);    		
    		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  
    		glwindow = glfwCreateWindow(800, 800, "example3", nullptr, nullptr);
    
    		int width, height;
			glfwGetFramebufferSize(glwindow, &width, &height);
			glViewport(0, 0, width, height);
    		glfwMakeContextCurrent(glwindow);
    
        	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    		glewExperimental = GL_TRUE;
    		// Initialize GLEW to setup the OpenGL Function pointers
    		glewInit();
    		
    		glEnable(GL_DEPTH_TEST);
    
    /*
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
    */
    
    //glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
    //glClear(GL_COLOR_BUFFER_BIT);
    
    /*
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(70,1,0.001*wdt/2.,1000*wdt/2.);
	glEnable(GL_DEPTH_TEST);//(NEW) Enable depth testing
	glDisable (GL_BLEND);
	*/

	const GLchar* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n"
    "layout (location = 1) in vec3 normal;\n"
    "layout (location = 2) in vec3 color;\n"
    "out vec3 Normal;\n"
	"out vec3 FragPos;\n"
	"out vec3 Color;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main()\n"
    "{\n"
    "gl_Position = projection * view * model * vec4(position, 1.0f);\n"
    "FragPos = vec3(model * vec4(position, 1.0f));\n"
    "Normal = mat3(transpose(inverse(model))) * normal;\n"
    "Color = color;\n"
    "}\0";
    /*
	const GLchar* fragmentShaderSource = "#version 330 core\n"
	"out vec4 color;\n"
  	"uniform vec3 objectColor;\n"
	"uniform vec3 lightColor;\n"
	"void main()\n"
	"{\n"
   	"color = vec4(lightColor * objectColor, 1.0f);\n"
	"}\0";
	*/
	/*
	const GLchar* fragmentShaderSource = "#version 330 core\n"
	"out vec4 color;\n"
  	"uniform vec3 objectColor;\n"
	"uniform vec3 lightColor;\n"
	"void main()\n"
	"{\n"
    "float ambientStrength = 0.1f;\n"
    "vec3 ambient = ambientStrength * lightColor;\n"
    "vec3 result = ambient * objectColor;\n"
    "color = vec4(result, 1.0f);\n"
	"}\0";
	*/
	const GLchar* fragmentShaderSource = "#version 330 core\n"
	"out vec4 color;\n"
	"in vec3 FragPos;\n" 
	"in vec3 Normal;\n"
	"in vec3 Color;\n"  
	"uniform vec3 lightPos;\n" 
	"uniform vec3 viewPos;\n"
	"uniform vec3 lightColor;\n"
	"void main()\n"
	"{\n"
    // Ambient
    "float ambientStrength = 0.1f;\n"
    "vec3 ambient = ambientStrength * lightColor;\n"
    // Diffuse 
    "vec3 norm = normalize(Normal);\n"
    "vec3 lightDir = normalize(lightPos - FragPos);\n"
    "float diff = abs(dot(norm, lightDir));\n"
    "vec3 diffuse = diff * lightColor;\n"   
    // Specular
    "float specularStrength = 0.5f;\n"
    "vec3 viewDir = normalize(viewPos - FragPos);\n"
    "vec3 reflectDir = reflect(-lightDir, norm);\n"  
    "float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n"
    "vec3 specular = specularStrength * spec * lightColor;\n"       
    "vec3 result = (ambient + diffuse + specular) * Color;\n"
    "color = vec4(result, 1.0f);\n"
	"}\0";
	
	const GLchar* lightvertexShaderSource = "#version 330 core\n"
	"layout (location = 0) in vec3 position;\n"
	"uniform mat4 model;\n"
	"uniform mat4 view;\n"
	"uniform mat4 projection;\n"
	"void main()\n"
	"{\n"
    "gl_Position = projection * view * model * vec4(position, 1.0f);\n"
	"}\0";

	const GLchar* lightfragmentShaderSource = "#version 330 core\n"
	"out vec4 color;\n"
 	"uniform vec3 objectColor;\n"
	"uniform vec3 lightColor;\n"
	"void main()\n"
	"{\n"
    "color = vec4(1.0f);\n"
	"}\0";

	
    // Build and compile our shader program
    // Vertex shader
    GLint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // Fragment shader
    GLint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // Vertex shader
    GLint lightvertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(lightvertexShader, 1, &lightvertexShaderSource, NULL);
    glCompileShader(lightvertexShader);
    // Check for compile time errors
    glGetShaderiv(lightvertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(lightvertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }    
    // Fragment shader
    GLint lightfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(lightfragmentShader, 1, &lightfragmentShaderSource, NULL);
    glCompileShader(lightfragmentShader);
    // Check for compile time errors
    glGetShaderiv(lightfragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(lightfragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }        
    // Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    lightshaderProgram = glCreateProgram();
    glAttachShader(lightshaderProgram, lightvertexShader);
    glAttachShader(lightshaderProgram, lightfragmentShader);
    glLinkProgram(lightshaderProgram);
    // Check for linking errors
    glGetProgramiv(lightshaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(lightshaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(lightvertexShader); 
    glDeleteShader(lightfragmentShader); 
    
    
    
    GLuint lightVAO;
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);
	// We only need to bind to the VBO, the container's VBO's data already contains the correct data.
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// Set the vertex attributes (only position data for our lamp)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

// Don't forget to 'use' the corresponding shader program first (to set the uniform)
GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
GLint lightColorLoc  = glGetUniformLocation(shaderProgram, "lightColor");
glUniform3f(objectColorLoc, 1.0f, 0.5f, 0.31f);
glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f); // Also set light's color (white)


        
    // Create nanogui gui
    bool enabled = true;
    
        // Create a nanogui screen and pass the glfw pointer to initialize
    screen = new nanogui::Screen();
    screen->initialize(glwindow, true);
    
    nanogui::FormHelper *gui = new nanogui::FormHelper(screen);
    nanogui::ref<nanogui::Window> nanoguiWindow = gui->addWindow(Eigen::Vector2i(10, 10), "");
    
    nanogui::Widget* tools = new nanogui::Widget(nanoguiWindow);
    tools->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
                                       nanogui::Alignment::Middle, 0, 6));
    nanogui::Button* b = new nanogui::Button(tools, "Load mesh");
    b->setCallback([&] {
     			if (active_project == NULL)
			std::cerr << "No active project" << std::endl;
				else{
					std::string str = nanogui::file_dialog(
                    {{"txt", "Mesh file"}}, false);
					std::vector<R3>  X;
					std::vector<N4>  Elts;
					std::vector<int> NbPts;
					std::cout << "Loading mesh file " << str << " ..." << std::endl;
					LoadMesh(str.c_str(),X,Elts,NbPts);
					GLMesh m(VBO,VAO,EBO,X,Elts,NbPts);
					set_mesh(m);
				}
    });
    
    b = new nanogui::Button(tools, "Load cluster");
    b->setCallback([&] {
     			if (active_project == NULL)
					std::cerr << "No active project" << std::endl;
				else if (active_project->get_mesh() == NULL)
					std::cerr << "No active mesh" << std::endl;
				else{
					std::string strmesh = nanogui::file_dialog(
                    {{"txt", "Mesh file"}}, false);
					std::cout << "Loading points from mesh file " << strmesh << " ..." << std::endl;
					vectReal r;
    				vectR3   x;
    				LoadPoints(strmesh.c_str(),x,r);					
					std::string strmat = nanogui::file_dialog(
                    {{"bin", "Matrix binary file"}}, false);
                    std::cout << "Loading matrix file " << strmat << " ..." << std::endl;              
    				Matrix   A;
                    bytes_to_matrix(strmat,A);
    				vectInt tab(nb_rows(A));
    				for (int j=0;j<x.size();j++){
    					tab[3*j]  = j;
        				tab[3*j+1]= j;
        				tab[3*j+2]= j;
    				}
					Cluster *t = new Cluster(x,r,tab);
					active_project->get_mesh()->set_cluster(t);
				}
    });    
    
    gui->addWidget("",tools);
    
    screen->setVisible(true);
    screen->performLayout();
    //nanoguiWindow->center();
    nanoguiWindow->setPosition(Eigen::Vector2i(600, 15));
    
    
        glfwSetCursorPosCallback(glwindow,
            [](GLFWwindow *, double x, double y) {
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
            	
            	
            screen->cursorPosCallbackEvent(x, y);
        }
    );

    glfwSetMouseButtonCallback(glwindow,
        [](GLFWwindow *, int button, int action, int modifiers) {
            screen->mouseButtonCallbackEvent(button, action, modifiers);
        }
    );

    glfwSetKeyCallback(glwindow,
        [](GLFWwindow *, int key, int scancode, int action, int mods) {
			GLMesh* mesh = NULL;
			if (active_project != NULL)
				mesh = active_project->get_mesh();
    		switch(key){
    			case 'A':case'a':
 						
        			break;
    			case 'D':case'd':

        			break;
    			case 'W':case'w':			
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
        		case 'Q':case'q':
        			if (action == GLFW_PRESS && mesh != NULL)
        				mesh->set_visudepth(std::max(0,(int)mesh->get_visudepth()-1));
        			break;
         		case 'E':case'e':
         			if (action == GLFW_PRESS && mesh != NULL)
        				mesh->set_visudepth(mesh->get_visudepth()+1);
        			break;
    			default:
        			break;
    		}
            screen->keyCallbackEvent(key, scancode, action, mods);
        }
    );

    glfwSetCharCallback(glwindow,
        [](GLFWwindow *, unsigned int codepoint) {
            screen->charCallbackEvent(codepoint);
        }
    );

    glfwSetDropCallback(glwindow,
        [](GLFWwindow *, int count, const char **filenames) {
            screen->dropCallbackEvent(count, filenames);
        }
    );

    glfwSetScrollCallback(glwindow,
        [](GLFWwindow *, double x, double y) {
            screen->scrollCallbackEvent(x, y);
       }
    );

    glfwSetFramebufferSizeCallback(glwindow,
        [](GLFWwindow *, int width, int height) {
            screen->resizeCallbackEvent(width, height);
        }
    );
			
			
			
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
			
			Project p("my project");
			projects.push_back(p);
			set_active_project(&(projects.front()));
			
/*			
		    glutInit(argc, argv);

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
			fbmesh->list->set_click_type(GLUI_SINGLE_CLICK);
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
*/			

			
		}
		
		void run() {
			while (!glfwWindowShouldClose(glwindow)) {
				glfwPollEvents();

       			draw();
		//draw();

        // Draw nanogui
        screen->drawContents();
        screen->drawWidgets();

        glfwSwapBuffers(glwindow);
    }
    
    glfwTerminate();
			// Loop require by OpenGL
			//glutMainLoop();
		}
		
};

std::list<Project> Scene::projects;
Project* Scene::active_project = NULL;
Real Scene::motionx, Scene::motiony;
nanogui::Screen* Scene::screen;
GLFWwindow* Scene::glwindow;
GLint Scene::shaderProgram, Scene::lightshaderProgram;
GLuint Scene::VBO, Scene::VAO, Scene::EBO;
}

#endif