# see https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py

import os
import pathlib
import re
import sys
import sysconfig
import platform
import subprocess
import shutil

from distutils.command.install_data import install_data
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts

class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info

    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        self.outfiles = self.distribution.data_files

class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True

        build_dir = self.build_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = [os.path.join(build_dir, _lib) for _lib in 
                os.listdir(build_dir) if 
                os.path.isfile(os.path.join(build_dir, _lib)) and 
                os.path.splitext(_lib)[1] in [".dll", ".so", ".dylib"]]
        

        for lib in libs:
            shutil.move(lib, os.path.join(self.build_dir,
                                          os.path.basename(lib)))

        # Mark the libs for installation, adding them to 
        # distribution.data_files seems to ensure that setuptools' record 
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list, 
        # but that never seemed to add them to the installed-files.txt in the 
        # egg-info, and the online recommendation seems to be adding libraries 
        # into eager_resources in the call to setup(), which I think puts them 
        # in data_files anyways. 
        # 
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [os.path.join(self.install_dir, 
                                                     os.path.basename(lib))
                                        for lib in libs]

        # Must be forced to run after adding the libs to data_files

        self.distribution.run_command("install_data")
        
        if (sys.version_info < (3, 0)):
            install_lib.run(self)
        else:
            super().run()


class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """
        The steps required to build the extension
        """
        
        self.announce("Preparing the build environment", level=3)

        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        if (sys.version_info < (3, 0)):
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            if not os.path.exists(str(build_temp.parent)):
                os.makedirs(str(extdir.parent))
        else:
            build_temp.mkdir(parents=True, exist_ok=True)
            extdir.mkdir(parents=True, exist_ok=True)

        # Now that the necessary directories are created, build

        self.announce("Configuring cmake project", level=3)

        # Change your cmake arguments below as necessary
        # os.chdir(str(build_temp))
        # print(os.getcwd())

        self.spawn(['cmake',str(cwd),
            '-B'+self.build_temp,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="+ str(extdir.parent.absolute()),
            "-DHTOOL_WITH_PYTHON_INTERFACE=True","-DCMAKE_BUILD_TYPE=Release"])

        self.spawn(['cmake', '--build', self.build_temp,"--target", "htool_shared","--config", "Release"])
        self.spawn(['cmake', '--build', self.build_temp,"--target","htool_shared_complex",
                    "--config", "Release"])
        # os.chdir(str(cwd))

        # cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]

        # cfg = 'Debug' if self.debug else 'Release'
        # build_args = ['--config', cfg]

        # if platform.system() == "Windows":
        #     cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
        #         cfg.upper(),
        #         extdir)]
        #     if sys.maxsize > 2**32:
        #         cmake_args += ['-A', 'x64']
        #     build_args += ['--', '/m']
        # else:
        #     cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        #     build_args += ['--', '-j2']

        # env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        #     env.get('CXXFLAGS', ''),
        #     self.distribution.get_version())
        # if not os.path.exists(self.build_temp):
        #     os.makedirs(self.build_temp)
        # subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
        #                       cwd=self.build_temp, env=env)
        # subprocess.check_call(['cmake', '--build', '.'] + build_args,
        #                       cwd=self.build_temp)

setup(
    name='Htool',
    version='1.0',
    package_dir = {'': 'interface'},
    # packages = {"htool"},
    # py_modules=['htool','htool.hmatrix'],
    ext_modules=[CMakeExtension('htool')],
    cmdclass={
          'build_ext': BuildCMakeExt,
          'install_data': InstallCMakeLibsData,
          'install_lib': InstallCMakeLibs,
        #   'install_scripts': InstallCMakeScripts
          },
    install_requires=[
          'numpy',
          'scipy',
          'mpi4py',
          'matplotlib'
      ],
    packages=setuptools.find_packages("interface"),
)
