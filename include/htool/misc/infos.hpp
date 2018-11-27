#ifndef HTOOL_INFOS_HPP
#define HTOOL_INFOS_HPP
#include <map>
#include <string>
#include <iostream>



namespace htool {

class Infos {
private:
  typedef std::map<std::string, double> inner_map;
  typedef std::map<std::string, inner_map> outer_map;

  static outer_map infos;
public:
  Infos();

  // Getters
  friend int Get_counter();
  friend const inner_map& Get_infos(const std::string& name);
  friend const double& Get_info(const std::string& name, const std::string& value_name);

  // Setters
  friend void Add_info(const std::string& name, const std::string& value_name, double value);
  friend void Add_value(const std::string& name, const std::string& value_name, double value);

  // Prints
  friend void Print_infos();
  friend void Print_infos(const std::string& name);

};

// Allocation de la mémoire pour les valeurs statiques (obligatoire)
Infos::outer_map Infos::infos;

// Getters
int Get_counter() {return Infos::infos.size();}
const Infos::inner_map& Get_infos(const std::string& name){return Infos::infos[name];}
const double& Get_info(const std::string& name,const std::string& value_name){
  return Infos::infos[name][value_name];
}

// Setters
void Add_info(const std::string& name, const std::string& value_name, double value){Infos::infos[name][value_name]=value;}
void Add_value(const std::string& name, const std::string& value_name, double value){Infos::infos[name][value_name]+=value;}

// Prints
void Print_infos(){
  for( auto i=Infos::infos.begin(); i!=Infos::infos.end(); ++i) {
    std::cout << i->first<< std::endl;
    for( auto j=i->second.begin(); j!=i->second.end(); ++j) {
      std::cout <<"\t"<<j->first<<" "<<j->second<<std::endl;
    }
   std::cout << std::endl;
   std::cout << std::endl;
  }

}


void Print_infos(const std::string& name){
  std::cout << name<< std::endl;
  for( auto j=Infos::infos[name].begin(); j!=Infos::infos[name].end(); ++j) {
    std::cout <<"\t"<<j->first<<" "<<j->second<<std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
}



Infos htool_infos();

}//namespace


#endif
