#ifndef HTOOL_VIEW_HPP
#define HTOOL_VIEW_HPP

#include <algorithm>
#include <list>
#include "../htool.hpp"

namespace htool {

  class Palette{
    public:
      unsigned int n;
      std::vector<R3> colors;
      Palette();
      Palette(const unsigned int nb, const std::vector<R3>& cols);
      Palette(const Palette& p);
      R3 get_color(float z) const;
  };

  Palette default_palette;

  Palette bw_palette;

class GLMesh;

class Camera{
  public:
    R3 eye,x,y,z,center,up;

    Camera();

    Camera(const R3& eye0, const R3& center0, const R3& up0);

    void set(const R3& eye0, const R3& center0, const R3& up0);

    void center_on(const GLMesh& mesh);
};

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
    Palette palette;
    std::shared_ptr<Cluster_tree> cluster;
    std::vector<int> tab;

  public:
    GLMesh(const GLMesh& m);

    ~GLMesh();

    GLMesh(const std::vector<R3>& X0, const std::vector<N4>&  Elts0, const std::vector<int>& NbPts0, const std::vector<R3>& normals0, const std::shared_ptr<Cluster_tree>& cluster0 = nullptr, const Palette& palette0 = default_palette);

    const R3& get_lbox() const;

    const R3& get_ubox() const;

    const unsigned int& get_visudepth() const;
    const std::shared_ptr<Cluster_tree> get_cluster() const;
    void set_cluster(const std::shared_ptr<Cluster_tree>& c);

    const std::vector<int>& get_tab() const;
    void set_tab(const std::vector<int>& t);

    void set_labels(std::vector<int>& l);
    void set_nblabels(unsigned int n);

    void set_palette(const Palette& p);

    void TraversalBuildLabel(const Cluster& t, std::vector<int>& labeldofs);
    void set_visudepth(const unsigned int depth);

    void set_buffers();

    void draw(const Camera& cam);
};

class Project{
  private:
    GLMesh* mesh;
    std::string name;
    Camera cam;
    IMatrix<K>* matrix;
    std::vector<R3>* ctrs;
    std::vector<double>* rays;
  public:
    Project(const char* s);

    ~Project();

    GLMesh* get_mesh() const;
    void set_mesh(const GLMesh& m);

    IMatrix<K>* get_matrix() const;
    void set_matrix(IMatrix<K>* m);

    std::vector<R3>* get_ctrs() const;
    void set_ctrs(const std::vector<R3>& m);

    std::vector<double>* get_rays() const;
    void set_rays(const std::vector<double>& m);

    std::string& get_name();
    Camera& get_camera();

    void set_camera(const Camera& c);
    void center_view_on_mesh();

    void draw();
};

class statics{
  public:
    std::list<Project> projects;
    Project* active_project;
    double motionx, motiony;
    nanogui::Screen* screen;
    GLFWwindow* glwindow;
    GLint shaderProgram, blackshaderProgram, lightshaderProgram;
    GLuint VBO, VAO, EBO;
    bool left_mouse_button_pressed;
};

class Scene{
  public:
    static statics gv;

    Scene();

    void set_active_project(Project* p);
    void set_mesh(const GLMesh& mesh);

    void init();

    void draw();

    void run();
};

statics Scene::gv;

Palette::Palette() {}

Palette::Palette(const unsigned int nb, const std::vector<R3>& cols) {
  n = nb;
  colors = cols;
}

Palette::Palette(const Palette& p) {
  n = p.n;
  colors = p.colors;
}

R3 Palette::get_color(float z) const{
  unsigned int i=(int)(z*(n-1));
  double t=z*(n-1)-i;
  //std::cout << z << " " << i << " " << t << std::endl;
  R3 col = ((1-t)*colors[i]+t*colors[std::min(n-1,i+1)])*(1./255);
  return col;
}

class MyGLCanvas : public nanogui::GLCanvas {
public:
    MyGLCanvas(Widget *parent, const HMatrix<partialACA,K>* A) : nanogui::GLCanvas(parent), mat(A) {
        using namespace nanogui;

        mShaderblocks.init(
            /* An identifying name */
            "a_simple_shader",

            /* Vertex shader */
            "#version 330\n"
            "uniform mat4 modelViewProj;\n"
            "in vec3 position;\n"
            "in vec3 color;\n"
            "out vec4 frag_color;\n"
            "void main() {\n"
            "    frag_color = vec4(color, 1.0);\n"
            "    gl_Position = modelViewProj * vec4(position, 1.0);\n"
            "}",

            /* Fragment shader */
            "#version 330\n"
            "out vec4 color;\n"
            "in vec4 frag_color;\n"
            "void main() {\n"
            "    color = frag_color;\n"
            "}"
        );

        mShaderwireframe.init(
            /* An identifying name */
            "a_simple_shader",

            /* Vertex shader */
            "#version 330\n"
            "uniform mat4 modelViewProj;\n"
            "in vec3 position;\n"
            "in vec3 color;\n"
            "out vec4 frag_color;\n"
            "void main() {\n"
            "    frag_color = vec4(color, 1.0);\n"
            "    gl_Position = modelViewProj * vec4(position, 1.0);\n"
            "}",

            /* Fragment shader */
            "#version 330\n"
            "out vec4 color;\n"
            "in vec4 frag_color;\n"
            "void main() {\n"
            "    color = frag_color;\n"
            "}"
        );

        const std::vector<partialACA<K>*>& lrmats = A->get_MyFarFieldMats();
        const std::vector<SubMatrix<K>*>& dmats = A->get_MyNearFieldMats();

        NbTri = 2*(dmats.size()+lrmats.size());
        NbSeg = 4*(dmats.size()+lrmats.size());

        MatrixXu indices(3, NbTri);
        MatrixXf positions(3, 3*NbTri);
        MatrixXf colors(3, 3*NbTri);

        MatrixXu indices_seg(2, NbSeg);
        MatrixXf positions_seg(3, 2*NbSeg);
        MatrixXf colors_seg(3, 2*NbSeg);

        int si = A->nb_rows();
        int sj = A->nb_cols();

        for (int i=0;i<dmats.size();i++) {
            const SubMatrix<K>& l = *(dmats[i]);
            indices.col(2*i) << 6*i, 6*i+1, 6*i+2;
            indices.col(2*i+1) << 6*i+3, 6*i+4, 6*i+5;
            positions.col(6*i) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*i+1) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*i+2) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions.col(6*i+3) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*i+4) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions.col(6*i+5) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            for (int j=0; j<6;j++)
              colors.col(6*i+j) << 1,0,0;

            for (int j=0; j<4;j++)
              indices_seg.col(4*i+j) << 8*i+2*j, 8*i+2*j+1;
            positions_seg.col(8*i) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*i+1) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*i+2) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*i+3) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*i+4) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*i+5) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*i+6) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*i+7) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;

            for (int j=0; j<8;j++)
              colors_seg.col(8*i+j) << 0,0,0;
        }

        for (int i=0;i<lrmats.size();i++) {
            const partialACA<K>& l = *(lrmats[i]);
            indices.col(2*dmats.size()+2*i) << 6*dmats.size()+6*i, 6*dmats.size()+6*i+1, 6*dmats.size()+6*i+2;
            indices.col(2*dmats.size()+2*i+1) << 6*dmats.size()+6*i+3, 6*dmats.size()+6*i+4, 6*dmats.size()+6*i+5;
            positions.col(6*dmats.size()+6*i) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*dmats.size()+6*i+1) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*dmats.size()+6*i+2) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions.col(6*dmats.size()+6*i+3) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions.col(6*dmats.size()+6*i+4) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions.col(6*dmats.size()+6*i+5) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            R3 col = bw_palette.get_color(l.compression());
            for (int j=0; j<6;j++)
              colors.col(6*dmats.size()+6*i+j) << col[0],col[1],col[2];

            for (int j=0; j<4;j++)
              indices_seg.col(4*dmats.size()+4*i+j) << 8*dmats.size()+8*i+2*j, 8*dmats.size()+8*i+2*j+1;
            positions_seg.col(8*dmats.size()+8*i) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+1) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+2) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)l.get_offset_j()/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+3) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+4) << (float)(l.get_offset_i()+l.nb_rows())/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+5) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+6) << (float)l.get_offset_i()/si, -(float)(l.get_offset_j()+l.nb_cols())/sj, 0;
            positions_seg.col(8*dmats.size()+8*i+7) << (float)l.get_offset_i()/si, -(float)l.get_offset_j()/sj, 0;

            for (int j=0; j<8;j++)
              colors_seg.col(8*dmats.size()+8*i+j) << 0,0,0;
        }

        mShaderblocks.bind();
        mShaderblocks.uploadIndices(indices);
        mShaderblocks.uploadAttrib("position", positions);
        mShaderblocks.uploadAttrib("color", colors);

        mShaderwireframe.bind();
        mShaderwireframe.uploadIndices(indices_seg);
        mShaderwireframe.uploadAttrib("position", positions_seg);
        mShaderwireframe.uploadAttrib("color", colors_seg);
    }

    ~MyGLCanvas() {
        mShaderblocks.free();
        mShaderwireframe.free();
    }

    virtual void drawGL() override {
        using namespace nanogui;

        float pixelRatio = Scene::gv.screen->pixelRatio();
        Vector2f screenSize = Scene::gv.screen->size().cast<float>();
        Vector2i positionInScreen = absolutePosition();

        Vector2i size = (mSize.cast<float>() * pixelRatio).cast<int>(),
        imagePosition = (Vector2f(positionInScreen[0],screenSize[1] - positionInScreen[1] -(float) mSize[1]) * pixelRatio).cast<int>();

        Matrix4f mvp;
        mvp.setIdentity();
        //float fTime = (float)glfwGetTime();

        //glViewport(-1,-1,2,2);

        Eigen::Affine3f transform(Eigen::Translation3f(Eigen::Vector3f(-0.5,0.5,0)));

        mvp = transform.matrix();/*Eigen::Matrix3f(Eigen::AngleAxisf(mRotation[0]*fTime, Vector3f::UnitX()) *
                                                   Eigen::AngleAxisf(mRotation[1]*fTime,  Vector3f::UnitY()) *
                                                   Eigen::AngleAxisf(mRotation[2]*fTime, Vector3f::UnitZ())) * 10.25f;*/

        Matrix4f sc;
        sc.setIdentity();
        sc.topLeftCorner<3,3>() = Eigen::Scaling((float)2,(float)2,(float)1);

        mvp = sc*mvp;

        mShaderblocks.bind();
        mShaderblocks.setUniform("modelViewProj", mvp);

        //glEnable(GL_DEPTH_TEST);
        //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        mShaderblocks.drawIndexed(GL_TRIANGLES, 0, NbTri);
        //glDisable(GL_DEPTH_TEST);

        mShaderwireframe.bind();
        mShaderwireframe.setUniform("modelViewProj", mvp);
        mShaderwireframe.drawIndexed(GL_LINES, 0, NbSeg);

        NVGcontext *ctx = Scene::gv.screen->nvgContext();

        const std::vector<partialACA<K>*>& lrmats = mat->get_MyFarFieldMats();

        int si = mat->nb_rows();
        int sj = mat->nb_cols();


        for (int i=0;i<lrmats.size();i++) {
            const partialACA<K>& l = *(lrmats[i]);
            float scaling = l.rank_of()  < 10 ? 1.3 : (l.rank_of()  < 100 ? 1. : 0.7);
            nvgFontSize(ctx, scaling*std::min((float)l.nb_rows()/si,(float)l.nb_cols()/sj)*std::min(mSize.x(),mSize.y()));
             nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
             nvgFillColor(ctx, Color(255, 192));
             nvgText(ctx, mPos.x()+(float)(l.get_offset_i()+0.5*l.nb_rows())/si*mSize.x(), mPos.y()+(float)(l.get_offset_j()+0.5*l.nb_cols())/sj*mSize.y(), NbrToStr(l.rank_of()).c_str(), NULL);
        }

    }

    virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers) override {
      if (down) {
        const std::vector<partialACA<K>*>& lrmats = mat->get_MyFarFieldMats();
        const std::vector<SubMatrix<K>*>& dmats = mat->get_MyNearFieldMats();

        int si = mat->nb_rows();
        int sj = mat->nb_cols();

        Eigen::Vector2f fp((float)(p[0]-mPos.x())/mSize.x(),(float)(p[1]-mPos.y())/mSize.y());

        int found = 0;

        Eigen::Vector2i offsets;
        Eigen::Vector2i dims;

        for (int i=0;i<dmats.size();i++) {
            const SubMatrix<K>& l = *(dmats[i]);
            if ((fp[0] >= (float)l.get_offset_i()/si) && (fp[0] <= (float)(l.get_offset_i()+l.nb_rows())/si))
            if ((fp[1] >= (float)l.get_offset_j()/sj) && (fp[1] <= (float)(l.get_offset_j()+l.nb_cols())/sj)) {
                std::cout << "Dense block of size " << l.nb_rows() << " x " << l.nb_cols() << " at (" << l.get_offset_i() << "," << l.get_offset_j() << ")" << std::endl;
                offsets = Eigen::Vector2i(l.get_offset_i(),l.get_offset_j());
                dims = Eigen::Vector2i(l.nb_rows(),l.nb_cols());
                found = 1;
                break;
            }
        }

        if (!found)
        for (int i=0;i<lrmats.size();i++) {
            const LowRankMatrix<K>& l = *(lrmats[i]);
            if ((fp[0] >= (float)l.get_offset_i()/si) && (fp[0] <= (float)(l.get_offset_i()+l.nb_rows())/si))
            if ((fp[1] >= (float)l.get_offset_j()/sj) && (fp[1] <= (float)(l.get_offset_j()+l.nb_cols())/sj)) {
                std::cout << "Low rank block of size " << l.nb_rows() << " x " << l.nb_cols() << " at (" << l.get_offset_i() << "," << l.get_offset_j() << "): rank = " << l.rank_of() << ", compression = " << l.compression() << std::endl;
                offsets = Eigen::Vector2i(l.get_offset_i(),l.get_offset_j());
                dims = Eigen::Vector2i(l.nb_rows(),l.nb_cols());
                break;
            }
        }

        int sizeg = Scene::gv.active_project->get_ctrs()->size();
        const std::vector<int>& tab = Scene::gv.active_project->get_mesh()->get_tab();
        const std::vector<int>& perm = Scene::gv.active_project->get_mesh()->get_cluster()->get_perm();

        std::vector<int> labs(sizeg);
        std::fill(labs.begin(), labs.end(), 0);
        for (int i=offsets[0]; i<offsets[0]+dims[0]; i++)
          labs[tab[perm[i]]] = 1;
        for (int j=offsets[1]; j<offsets[1]+dims[1]; j++){
          if (labs[tab[perm[j]]] == 1)
            labs[tab[perm[j]]] = 3;
          else if (labs[tab[perm[j]]] == 0)
            labs[tab[perm[j]]] = 2;
          }

        Scene::gv.active_project->get_mesh()->set_labels(labs);
        Scene::gv.active_project->get_mesh()->set_nblabels(4);

        Palette block_palette;
        std::vector<R3> colorsblock(4);
        colorsblock[0][0]=255;colorsblock[0][1]=0;colorsblock[0][2]=0;
        colorsblock[1][0]=0;colorsblock[1][1]=0;colorsblock[1][2]=0;
        colorsblock[2][0]=255;colorsblock[2][1]=255;colorsblock[2][2]=255;
        colorsblock[3][0]=127;colorsblock[3][1]=127;colorsblock[3][2]=127;
        block_palette.n = 4;
        block_palette.colors = colorsblock;
        Scene::gv.active_project->get_mesh()->set_palette(block_palette);

        Scene::gv.active_project->get_mesh()->set_buffers();
      }
      return true;
    }

private:
    const HMatrix<partialACA,K>* mat;
    int NbTri, NbSeg;
    nanogui::GLShader mShaderblocks;
    nanogui::GLShader mShaderwireframe;
};

void LoadMesh(std::string inputname, std::vector<R3>&  X, std::vector<N4>&  Elt, std::vector<int>& NbPt, std::vector<R3>& Normals, std::vector<R3>& ctrs, std::vector<double>& rays) {
int   num,NbElt,poubelle, NbTri, NbQuad;
R3    Pt, Ctr;
double Rmax, Rad;
R3 v1,v2;

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
Normals.resize(NbElt);

num=0; NbTri=0; NbQuad=0;
// Lecture elements
for(int e=0; e<NbElt; e++){
  infile >> poubelle;
  infile >> NbPt[e];

  Ctr.fill(0);

  if(NbPt[e]==3){NbTri++;}
  if(NbPt[e]==4){NbQuad++;}

  // Calcul centre element
  for(int j=0; j<NbPt[e]; j++){
    infile >> poubelle;
    infile >> Pt;
    Elt[e][j] = num;
    X.push_back(Pt);
    num++;
    Ctr += (1./double(NbPt[e]))*Pt;
  }

  v1 = X[Elt[e][1]]-X[Elt[e][0]];
  v2 = X[Elt[e][2]]-X[Elt[e][0]];
  Normals[e] = v1^v2;
  Normals[e] = Normals[e]*(1./norm2(Normals[e]));

  ctrs.push_back(Ctr);
  Rmax = norm2(Ctr-X[Elt[e][0]]);

      for(int j=1; j<NbPt[e]; j++){
        Rad = norm2(Ctr-X[Elt[e][j]]);
    if (Rad > Rmax)
            Rmax=Rad;
      }

      rays.push_back(Rmax);

  // Separateur inter-element
  if(e<NbElt-1){infile >> poubelle;}

}

std::cout << NbTri << " triangle(s) and " << NbQuad << " quad(s)" << std::endl;

infile.close();
}

GLMesh::GLMesh(const GLMesh& m) : palette(m.palette){
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
  tab = m.tab;
}

GLMesh::~GLMesh() {
  X.clear();
  Elts.clear();
  NbPts.clear();
  normals.clear();
  labels.clear();
  tab.clear();
  //delete cluster;
}

GLMesh::GLMesh(const std::vector<R3>& X0, const std::vector<N4>&  Elts0, const std::vector<int>& NbPts0, const std::vector<R3>& normals0, const std::shared_ptr<Cluster_tree>& cluster0, const Palette& palette0)
  : palette(palette0), cluster(cluster0){
  X = X0;
  Elts = Elts0;
  NbPts = NbPts0;
  normals = normals0;
  lbox.fill(1.e+30);
  ubox.fill(-1.e+30);
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
  for (int i=0; i<Elts.size(); i++)
    labels[i] = 0;

  nblabels = 1;
  visudepth = 0;
}

const R3& GLMesh::get_lbox() const{
  return lbox;
}

const R3& GLMesh::get_ubox() const{
  return ubox;
}

const unsigned int& GLMesh::get_visudepth() const{
  return visudepth;
}

const std::shared_ptr<Cluster_tree> GLMesh::get_cluster() const{
  return cluster;
}

void GLMesh::set_cluster(const std::shared_ptr<Cluster_tree>& c) {
  cluster = c;
}

const std::vector<int>& GLMesh::get_tab() const{
  return tab;
}

void GLMesh::set_tab(const std::vector<int>& t) {
  tab = t;
}

void GLMesh::set_labels(std::vector<int>& l) {
  labels = l;
}

void GLMesh::set_nblabels(unsigned int n) {
  nblabels = n;
}

void GLMesh::set_palette(const Palette& p) {
  palette = p;
}

void GLMesh::TraversalBuildLabel(const Cluster& t, std::vector<int>& labeldofs){
  if(t.get_depth()<visudepth && !t.IsLeaf()){
    TraversalBuildLabel(t.get_son(0), labeldofs);
    TraversalBuildLabel(t.get_son(1), labeldofs);
  }
  else {
    /*
    for(int i=0; i<num_(t).size(); i++)
      labeldofs[num_(t)[i]] = nblabels;
    */

    for(int i=t.get_offset(); i<t.get_offset()+t.get_size(); i++)
      labeldofs[tab[cluster->get_perm()[i]]] = nblabels;

    nblabels++;
  }
}

void GLMesh::set_visudepth(const unsigned int depth){
  if (cluster == NULL)
    std::cerr << "No cluster for this GLMesh" << std::endl;
  else {
    visudepth = depth;
    nblabels = 1;
    int sizeg = Scene::gv.active_project->get_ctrs()->size();
    std::vector<int> labeldofs(sizeg);
    TraversalBuildLabel(cluster->get_root(), labeldofs);
    // P0
    // if (sizeg == Elts.size())
      labels = labeldofs;
    // else {
    //   for (int i=0; i<Elts.size(); i++) {
    //     std::map<int,int> m;
    //     for (int j=0; j<NbPts[i]; j++) {
    //       auto search = m.find(labeldofs[Elts[i][j]]);
    //         if(search != m.end()){
    //           search->second++;
    //       }
    //         else{
    //           m[labeldofs[Elts[i][j]]] = 1;
    //       }
    //     }
    //     auto max = std::max_element(m.begin(), m.end(),
    // [](const std::pair<int,int>& p1, const std::pair<int,int>& p2) {
    //     return p1.second < p2.second; });
    //     labels[i] = max->first;
    //   }
    // }
    set_buffers();
    std::cout << "Depth set to " << depth << std::endl;
  }
}

void GLMesh::set_buffers() {
  int np = NbPts[0];
  int sz = (np == 3 ? np : 6);
  R3 col;
  GLfloat* vertices= new GLfloat[9*sz*Elts.size()];
  for (int i=0; i<9*sz*Elts.size(); i++){
    vertices[i] = 0;
  }
  if (np == 3) {
    for (int i=0; i<Elts.size(); i++) {
      for (int j=0; j<3; j++){
        if (labels.size()==Elts.size())
          col = palette.get_color(1.*(labels[i])/(nblabels > 1 ? nblabels-1 : 1));
        else
          col = palette.get_color(1.*(labels[Elts[i][j]])/(nblabels > 1 ? nblabels-1 : 1));
        for (int k=0; k<3; k++) {
          vertices[9*sz*i+9*j+k] = X[Elts[i][j]][k];
          // Normals
          vertices[9*sz*i+9*j+3+k] = normals[i][k];
          // Colors
          vertices[9*sz*i+9*j+6+k] =col[k];
        }
      }
    }
  }
  else if (np == 4) {
    for (int i=0; i<Elts.size(); i++) {
      //R3 col = palette.get_color(1.*(labels[i]%32)/32);
      col = palette.get_color(1.*(labels[i])/(nblabels > 1 ? nblabels-1 : 1));
      for (int k=0; k<3; k++) {
        vertices[9*sz*i+9*0+k] = X[Elts[i][0]][k];
        vertices[9*sz*i+9*1+k] = X[Elts[i][1]][k];
        vertices[9*sz*i+9*2+k] = X[Elts[i][2]][k];
        vertices[9*sz*i+9*3+k] = X[Elts[i][0]][k];
        vertices[9*sz*i+9*4+k] = X[Elts[i][3]][k];
        vertices[9*sz*i+9*5+k] = X[Elts[i][2]][k];
        for (int j=0; j<6; j++) {
          // Normals
          vertices[9*sz*i+9*j+3+k] = normals[i][k];
          // Colors
          vertices[9*sz*i+9*j+6+k] =col[k];
        }
      }
    }
  }

  /*
  // Coordinates
  for (int i=0; i<X.size(); i++)
  for (int j=0; j<3; j++)
    vertices[9*i+j] = X[i][j];

  for (int i=0; i<Elts.size(); i++) {
    R3 col = palette.get_color(1.*(labels[i]%32)/32);
    for (int j=0; j<NbPts[i]; j++) {
      // Normals
      for (int k=0; k<3; k++)
        vertices[9*Elts[i][j]+3+k] += normals[i][k];
      // Colors
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
  */

  GLuint &VBO = Scene::gv.VBO;
  GLuint &VAO = Scene::gv.VAO;
  GLuint &EBO = Scene::gv.EBO;

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  //glGenBuffers(1, &EBO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, 9*sz*Elts.size() * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

  glBindVertexArray(VAO);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);
  // Normal attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
  glEnableVertexAttribArray(1);
  // Color attribute
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
  glEnableVertexAttribArray(2);

  /*
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);
  */
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  delete [] vertices;
}

void GLMesh::draw(const Camera& cam) {
  glEnable(GL_DEPTH_TEST);

  R3 lp = cam.eye;
  glm::vec3 lightPos(lp[0],lp[1],lp[2]);

  glm::mat4 view = glm::lookAt(glm::vec3(cam.eye[0],cam.eye[1],cam.eye[2]), glm::vec3(cam.center[0],cam.center[1],cam.center[2]), glm::vec3(cam.up[0],cam.up[1],cam.up[2]));
  float wdt = 100;
  //glm::mat4 projection = glm::perspective(0.f, 1.f, 0.1f, 1000.0f);
  //glm::mat4 projection = glm::perspective(70.f, 1.f, 0.001f, 100.0f);
  glm::mat4 projection = glm::perspective(70.f,1.f,(float)(0.001*wdt/2.),(float)(1000*wdt/2.));
  //70,1,0.001*wdt/2.,1000*wdt/2.

  GLint shaderProgram = Scene::gv.shaderProgram;
    glUseProgram(shaderProgram);

  GLint lightColorLoc  = glGetUniformLocation(shaderProgram, "lightColor");
  GLint lightPosLoc    = glGetUniformLocation(shaderProgram, "lightPos");
  GLint viewPosLoc     = glGetUniformLocation(shaderProgram, "viewPos");
  glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f);
  glUniform3f(lightPosLoc,    lightPos.x, lightPos.y, lightPos.z);
  glUniform3f(viewPosLoc,     cam.eye[0],cam.eye[1],cam.eye[2]);
  GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
  GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
  GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

  GLuint &VAO = Scene::gv.VAO;

  glBindVertexArray(VAO);

  glm::mat4 model;
  //model = glm::translate(model, glm::vec3(0.f, 0.f, 0.f));
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  if (NbPts[0] == 3)
    glDrawArrays(GL_TRIANGLES, 0, 3*Elts.size());
  else
    glDrawArrays(GL_TRIANGLES, 0, 6*Elts.size());

  /*
  if (NbPts[0] == 3)
    glDrawElements(GL_TRIANGLES, 3*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
  else
    glDrawElements(GL_TRIANGLES, 6*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
  */

  GLint blackshaderProgram = Scene::gv.blackshaderProgram;
    glUseProgram(blackshaderProgram);

  lightColorLoc  = glGetUniformLocation(blackshaderProgram, "lightColor");
  lightPosLoc    = glGetUniformLocation(blackshaderProgram, "lightPos");
  viewPosLoc     = glGetUniformLocation(blackshaderProgram, "viewPos");
  glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f);
  glUniform3f(lightPosLoc,    lightPos.x, lightPos.y, lightPos.z);
  glUniform3f(viewPosLoc,     cam.eye[0],cam.eye[1],cam.eye[2]);
  modelLoc = glGetUniformLocation(blackshaderProgram, "model");
  viewLoc = glGetUniformLocation(blackshaderProgram, "view");
  projLoc = glGetUniformLocation(blackshaderProgram, "projection");
  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  /*
  if (NbPts[0] == 3)
    glDrawElements(GL_TRIANGLES, 3*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
  else
    glDrawElements(GL_TRIANGLES, 6*Elts.size(), GL_UNSIGNED_INT, (void*)0 );
  */
  if (NbPts[0] == 3)
    glDrawArrays(GL_TRIANGLES, 0, 3*Elts.size());
  else
    glDrawArrays(GL_TRIANGLES, 0, 6*Elts.size());

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glBindVertexArray(0);
}

Camera::Camera() {
  eye.fill(0);
  center.fill(1);
  up.fill(1);
  x.fill(1);
  y.fill(0);
  z.fill(0);
}

Camera::Camera(const R3& eye0, const R3& center0, const R3& up0){
  eye = eye0;
  center = center0;
  up = up0;
  z = center-eye;
  y = up;
  z /= norm2(z);
  y /= norm2(y);
  x = y^z;
  x /= norm2(x);
}

void Camera::set(const R3& eye0, const R3& center0, const R3& up0){
  eye = eye0;
  center = center0;
  up = up0;
  z = center-eye;
  y = up;
  z /= norm2(z);
  y /= norm2(y);
  x = y^z;
  x /= norm2(x);
}

void Camera::center_on(const GLMesh& mesh){
  center = 0.5*(mesh.get_lbox()+mesh.get_ubox());
  up.fill(0);
  up[2] = 1;
  eye = center-2.*(mesh.get_ubox()-mesh.get_lbox());
  eye[2] = 0;
  z = center-eye;
  y = up;
  z /= norm2(z);
  y /= norm2(y);
  x = y^z;
  x /= norm2(x);
}

Project::Project(const char* s){
  mesh = NULL;
  matrix = NULL;
  ctrs = NULL;
  rays = NULL;
  name = s;
}

Project::~Project(){
  if (mesh != NULL) {
    delete mesh;
    mesh = NULL;
  }
  if (matrix != NULL) {
    delete matrix;
    matrix = NULL;
  }
  if (ctrs != NULL) {
    delete ctrs;
    ctrs = NULL;
  }
  if (rays != NULL) {
    delete rays;
    rays = NULL;
  }
}

GLMesh* Project::get_mesh() const{
  return mesh;
}

void Project::set_mesh(const GLMesh& m) {
  if (mesh != NULL)
    delete mesh;
  mesh = new GLMesh(m);
}

IMatrix<K>* Project::get_matrix() const{
  return matrix;
}

void Project::set_matrix(IMatrix<K>* m) {
  if (matrix != NULL)
    delete matrix;
  matrix = m;
}

std::vector<R3>* Project::get_ctrs() const{
  return ctrs;
}

void Project::set_ctrs(const std::vector<R3>& m) {
  if (ctrs != NULL)
    delete ctrs;
  ctrs = new std::vector<R3>(m);
}

std::vector<double>* Project::get_rays() const{
  return rays;
}

void Project::set_rays(const std::vector<double>& m) {
  if (rays != NULL)
    delete rays;
  rays = new std::vector<double>(m);
}


std::string& Project::get_name() {
  return name;
}

Camera& Project::get_camera() {
  return cam;
}

void Project::set_camera(const Camera& c){
  cam = c;
}

void Project::center_view_on_mesh() {
  if (mesh != NULL)
    cam.center_on(*mesh);
}

void Project::draw() {
  if (mesh != NULL)
    mesh->draw(cam);
}

Scene::Scene() {}

void Scene::set_active_project(Project* p) {
  gv.active_project = p;
}

void Scene::set_mesh(const GLMesh& mesh){
  if (gv.active_project == NULL)
    std::cerr << "No active project" << std::endl;
  else {
    gv.active_project->set_mesh(mesh);
    gv.active_project->get_mesh()->set_buffers();
    gv.active_project->center_view_on_mesh();
  }
}

void Scene::draw(){
  glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (gv.active_project != NULL)
    gv.active_project->draw();
}

void Scene::init(){
  glfwInit();
  glfwSetTime(0);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  gv.glwindow = glfwCreateWindow(1280, 1024, "htool gui", nullptr, nullptr);

  int width, height;
  glfwGetFramebufferSize(gv.glwindow, &width, &height);
  glViewport(0, 0, width, height);
  glfwMakeContextCurrent(gv.glwindow);

  // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
  //glewExperimental = GL_TRUE;
  // Initialize GLEW to setup the OpenGL Function pointers
  //glewInit();

  glEnable(GL_DEPTH_TEST);

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

  const GLchar* blackvertexShaderSource = "#version 330 core\n"
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
  "vec4 v = view * model * vec4(position, 1.0f);\n"
  "v.xyz = v.xyz * 0.995;\n"
  "gl_Position = projection * v;\n"
  "FragPos = vec3(model * vec4(position, 1.0f));\n"
  "FragPos = FragPos * 0.995;\n"
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

  const GLchar* blackfragmentShaderSource = "#version 330 core\n"
  "out vec4 color;\n"
  "in vec3 FragPos;\n"
  "in vec3 Normal;\n"
  "uniform vec3 lightPos;\n"
  "uniform vec3 viewPos;\n"
  "uniform vec3 lightColor;\n"
  "void main()\n"
  "{\n"
  "vec3 Color = vec3(0.0f,0.0f,0.0f);\n"
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

   // black Vertex shader
   GLint blackvertexShader = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(blackvertexShader, 1, &blackvertexShaderSource, NULL);
   glCompileShader(blackvertexShader);
   // Check for compile time errors
   glGetShaderiv(blackvertexShader, GL_COMPILE_STATUS, &success);
   if (!success)
   {
       glGetShaderInfoLog(blackvertexShader, 512, NULL, infoLog);
       std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
   }

   // black Fragment shader
   GLint blackfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(blackfragmentShader, 1, &blackfragmentShaderSource, NULL);
   glCompileShader(blackfragmentShader);
   // Check for compile time errors
   glGetShaderiv(blackfragmentShader, GL_COMPILE_STATUS, &success);
   if (!success)
   {
       glGetShaderInfoLog(blackfragmentShader, 512, NULL, infoLog);
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

   GLint& shaderProgram = gv.shaderProgram;
   GLint& lightshaderProgram = gv.lightshaderProgram;
   GLint& blackshaderProgram = gv.blackshaderProgram;

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

   blackshaderProgram = glCreateProgram();
   glAttachShader(blackshaderProgram, blackvertexShader);
   glAttachShader(blackshaderProgram, blackfragmentShader);
   glLinkProgram(blackshaderProgram);
   // Check for linking errors
   glGetProgramiv(blackshaderProgram, GL_LINK_STATUS, &success);
   if (!success) {
       glGetProgramInfoLog(blackshaderProgram, 512, NULL, infoLog);
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
   glDeleteShader(blackvertexShader);
   glDeleteShader(blackfragmentShader);
   glDeleteShader(lightvertexShader);
   glDeleteShader(lightfragmentShader);



   /*
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
*/


  // Create nanogui gui
  bool enabled = true;

  // Create a nanogui screen and pass the glfw pointer to initialize

  gv.screen = new nanogui::Screen();
  gv.screen->initialize(gv.glwindow, true);

  nanogui::FormHelper *gui = new nanogui::FormHelper(gv.screen);
  nanogui::ref<nanogui::Window> nanoguiWindow = gui->addWindow(Eigen::Vector2i(10, 10), "");

   gui->addGroup("Hmatrix parameters");
   gui->addVariable("eta", Parametres::eta)->setSpinnable(true);
   gui->addVariable("epsilon", Parametres::epsilon)->setSpinnable(true);
   gui->addVariable("max block size", Parametres::maxblocksize)->setSpinnable(true);
   gui->addVariable("min cluster size", Parametres::minclustersize)->setSpinnable(true);
   gui->addButton("Compute hmatrix",[&]{
         if (gv.active_project == NULL)
      std::cerr << "No active project" << std::endl;
    else if (gv.active_project->get_mesh() == NULL)
      std::cerr << "No mesh loaded" << std::endl;
    else if (gv.active_project->get_matrix() == NULL)
      std::cerr << "No matrix loaded" << std::endl;
    else {
      IMatrix<K>& A = *(gv.active_project->get_matrix());
      const std::vector<R3>& x = *(gv.active_project->get_ctrs());
      const std::vector<int>& tab = gv.active_project->get_mesh()->get_tab();
      //const std::vector<double>& r = *(gv.active_project->get_rays());
      std::shared_ptr<Cluster_tree> t=std::make_shared<Cluster_tree>(x,tab);
      gv.active_project->get_mesh()->set_cluster(t);
      // HMatrix<partialACA,K> pB(A,t,x,tab);
      HMatrix<partialACA,K>* B = new HMatrix<partialACA,K>(A,t,x,tab);

      /*
      vectCplx ua(nr),ub(nr);

      MvProd(ua,A,u);
      std::pair <double,double > mvp_stats= MvProdMPI(ub,B,u);

      add_stats(B,"MvProd (mean)",std::get<0>(mvp_stats));
      add_stats(B,"MvProd (max)",std::get<1>(mvp_stats));
      add_stats(B,"MvProd err",norm(ua-ub)/norm(ua));
      */
      //Real normA = NormFrob(A);

      B->add_info("Compression",NbrToStr(B->compression()));
      B->add_info("Nb dense mats",NbrToStr(B->get_ndmat()));
      B->add_info("Nb lr mats",NbrToStr(B->get_nlrmat()));
      //add_stats(B,"Relative Frob error",sqrt(squared_absolute_error(B,A))/normA);

      nanogui::Window *popup = new nanogui::Window(Scene::gv.screen, "Stats");

      popup->setPosition(Eigen::Vector2i(350, 250));
      popup->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
      nanogui::Alignment::Middle, 10, 10));
      nanogui::Widget *panel1 = new nanogui::Widget(popup);
      panel1->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal,
      nanogui::Alignment::Middle, 10, 15));

      MyGLCanvas* mCanvas = new MyGLCanvas(panel1,B);
      mCanvas->setSize(Eigen::Vector2i(300,300));

      const std::map<std::string,std::string>& stats = B->get_infos();
      std::stringstream s;

      s << "eta" << "\t" << Parametres::eta << "\n";
      s << "epsilon" << "\t" << Parametres::epsilon << "\n";
      s << "max block size" << "\t" << Parametres::maxblocksize << "\n";
      s << "min cluster size" << "\t" << Parametres::minclustersize << "\n";
      s << "\n";
      for (auto it = stats.begin() ; it != stats.end() ; ++it){
        if (it->first.find("mean") == std::string::npos)
        s << it->first << "\t" << it->second << "\n";
      }

      nanogui::Label *mMessageLabel = new nanogui::Label(panel1, s.str().c_str());
      mMessageLabel->setFixedWidth(200);
      nanogui::Widget *panel2 = new nanogui::Widget(popup);
      panel2->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal,
      nanogui::Alignment::Middle, 0, 15));

      nanogui::Button *b = new nanogui::Button(panel2, "Close", ENTYPO_ICON_CHECK);
      b->setCallback([popup] {
        popup->dispose();
        /*
        panel1->setFixedSize(Eigen::Vector2i(900,900));
        panel1->setSize(Eigen::Vector2i(900,900));
        panel1->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,nanogui::Alignment::Maximum, 10, 15));
        Scene::gv.screen->performLayout();
        */
      });

      nanogui::Button *bf = new nanogui::Button(panel2, "FullScreen", ENTYPO_ICON_CHECK);
      bf->setFlags(nanogui::Button::ToggleButton);
      bf->setChangeCallback([popup,mCanvas,panel1,mMessageLabel](bool state) {
        if (state) {
          mCanvas->setSize(Scene::gv.screen->size()-Eigen::Vector2i(0,60));
          popup->setPosition(Eigen::Vector2i(0,0));
          popup->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
          nanogui::Alignment::Middle, 0,0));
          panel1->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal,
          nanogui::Alignment::Middle, 0,0));
          mMessageLabel->setVisible(0);
          Scene::gv.screen->performLayout();
        }
        else {
          mCanvas->setSize(Eigen::Vector2i(300,300));
          popup->setPosition(Eigen::Vector2i(350, 250));
          popup->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical,
          nanogui::Alignment::Middle, 10, 10));
          panel1->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal,
          nanogui::Alignment::Middle, 10, 15));
          mMessageLabel->setVisible(1);
          Scene::gv.screen->performLayout();
        }
      });

      Scene::gv.screen->performLayout();

/*
          nanogui::Popup *popup = popupBtn->popup();
        popup->setLayout(new nanogui::GroupLayout());
        */
      /*
      auto dlg = new nanogui::MessageDialog(Scene::gv.screen, nanogui::MessageDialog::Type::Information, "Title", "This is an information message");
            dlg->setCallback([](int result) { std::cout << "Dialog result: " << result << std::endl; });
            */


     }
   });

   gv.screen->setVisible(true);
   gv.screen->performLayout();
   //nanoguiWindow->center();
   nanoguiWindow->setPosition(Eigen::Vector2i(550, 15));


       glfwSetCursorPosCallback(gv.glwindow,
           [](GLFWwindow *, double x, double y) {
    if (gv.active_project != NULL && gv.left_mouse_button_pressed == true){

    Camera& cam = gv.active_project->get_camera();
    R3 trans = cam.center;
    cam.center.fill(0);
    cam.eye = cam.eye - trans;
    double scale = norm2(cam.eye);
    cam.eye /= scale;

    if (x != gv.motionx) {
           float depl = 0.1;
           if (x > gv.motionx)
                 depl *= -1;
           float xx = cam.eye[0]*cos(depl) - cam.eye[1]*sin(depl);
           float yy = cam.eye[0]*sin(depl) + cam.eye[1]*cos(depl);
           cam.eye[0] = xx;
           cam.eye[1] = yy;
      cam.center.fill(0);

      R3 zz;
      zz.fill(0);
      zz[2] = cam.y[2];
      zz /= norm2(zz);
      cam.z = cam.center-cam.eye;
      cam.z /= norm2(cam.z);
      cam.x = zz^cam.z;
      cam.x /= norm2(cam.x);
      cam.y = cam.x^cam.z;
      cam.y *= -1.;
      cam.y /= norm2(cam.y);

      cam.center = cam.eye+cam.z;
      cam.up = cam.y;
    }

    if (y != gv.motiony) {
      float depl = 0.1;
      R3 oldeye = cam.eye;
      if (y > gv.motiony)
        depl *= -1;
      R3 uu = cam.y;
      uu /= (double)-tan(std::abs(depl)/2);
      uu += cam.z;
      uu /= norm2(uu);
      double nor = 2*norm2(cam.eye)*sin(depl/2);
      uu = nor*uu;
      cam.eye += uu;
      cam.center.fill(0);
      nor = norm2(oldeye);
      cam.eye /= norm2(cam.eye);
      cam.eye = nor*cam.eye;
      cam.z = cam.center-cam.eye;
      cam.z /= norm2(cam.z);
      cam.y = cam.x^cam.z;
      cam.y *= -1.;
      cam.y /= norm2(cam.y);

      cam.center = cam.eye+cam.z;
      cam.up = cam.y;
    }

    cam.eye = scale*cam.eye;

    cam.center += trans;
    cam.eye += trans;

    gv.motionx = x;
    gv.motiony = y;
    }

         gv.screen->cursorPosCallbackEvent(x, y);
       }
   );

   glfwSetMouseButtonCallback(gv.glwindow,
       [](GLFWwindow *, int button, int action, int modifiers) {
      if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        gv.left_mouse_button_pressed = true;
      if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        gv.left_mouse_button_pressed = false;
      gv.screen->mouseButtonCallbackEvent(button, action, modifiers);
       }
   );

   glfwSetKeyCallback(gv.glwindow,
       [](GLFWwindow *, int key, int scancode, int action, int mods) {
    GLMesh* mesh = NULL;
    if (gv.active_project != NULL)
      mesh = gv.active_project->get_mesh();
       switch(key){
         case 'A':case'a':

             break;
         case 'D':case'd':

             break;
         case 'W':case'w':
            if (gv.active_project != NULL){
             Camera& cam = gv.active_project->get_camera();
             cam.eye += 0.15*(cam.center-cam.eye);
           }
             break;
         case 'S':case's':
            if (gv.active_project != NULL){
             Camera& cam = gv.active_project->get_camera();
             cam.eye += -0.15*(cam.center-cam.eye);
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
           gv.screen->keyCallbackEvent(key, scancode, action, mods);
       }
   );

   glfwSetCharCallback(gv.glwindow,
       [](GLFWwindow *, unsigned int codepoint) {
           gv.screen->charCallbackEvent(codepoint);
       }
   );

   glfwSetDropCallback(gv.glwindow,
       [](GLFWwindow *, int count, const char **filenames) {
           gv.screen->dropCallbackEvent(count, filenames);
       }
   );

   glfwSetScrollCallback(gv.glwindow,
       [](GLFWwindow *, double x, double y) {
           gv.screen->scrollCallbackEvent(x, y);
      }
   );

   glfwSetFramebufferSizeCallback(gv.glwindow,
       [](GLFWwindow *, int width, int height) {
           gv.screen->resizeCallbackEvent(width, height);
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

    std::vector<R3> colorsbw(2);
    colorsbw[0][0]=20;colorsbw[0][1]=80;colorsbw[0][2]=20;
    colorsbw[1][0]=100;colorsbw[1][1]=250;colorsbw[1][2]=100;
    bw_palette.n = 2;
    bw_palette.colors = colorsbw;

    /*
    std::vector<R3> colors(3);
    colors[0][0]=59;colors[0][1]=76;colors[0][2]=192;
    colors[1][0]=221;colors[1][1]=221;colors[1][2]=221;
    colors[2][0]=0.916482116*255;colors[2][1]=0.236630659*255;colors[2][2]=0.209939162*255;
    default_palette.n = 3;
    default_palette.colors = colors;
    */

    Project p("my project");
    gv.projects.push_back(p);
    set_active_project(&(gv.projects.front()));

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

void Scene::run() {
  while (!glfwWindowShouldClose(gv.glwindow)) {
    glfwPollEvents();

    draw();

    // Draw nanogui
    gv.screen->drawContents();
    gv.screen->drawWidgets();

    glfwSwapBuffers(gv.glwindow);
  }

   glfwTerminate();
}

}

#endif
