Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

h = 0.07;
R = 1;

Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
Disk(1) = {0,0,0,R};
