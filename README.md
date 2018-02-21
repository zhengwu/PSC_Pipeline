# Scilpy

**Scilpy** is the main internal library supporting research and developement at the Sherbrooke Connectivity Imaging Lab
([SCIL] [1]).

The library's structure is mostly aligned on that of [Dipy].

By default, **Scilpy** is only accessible to active members of the lab and some collaborators. This is to avoid people
using code which is still unstable and in development. Once a subpart has matured enough, or has been published, we
encourage members of the lab to contribute this part to [Dipy].

A readme file about various registration scripts is available in doc/tractogram_registration.md, it contains
information on how to use the scripts (individually or one after the other)

### Dependencies and installation

On Debian/Ubuntu and derivatives, you can get some dependencies from the repositories by running

sudo apt-get install python-numpy python-scipy python-pip libgsl0-dev

The rest of the dependencies can be installed by running at the root of the folder

pip install -r requirements.txt


### Cythonized extensions

Some parts of the library have been Cythonized to gain some significant speed improvements. To be able to use those
sped-up parts, you will need to call the setup.py script once you have obtained the source code. You have 2 choices:

* If you have all GSL dependencies installed, you can build everything by calling
```
python setup.py build_all
```

* Else, or if you do not need the parts depending on GSL, you can call
```
python setup.py build_no_gsl
```

### What happens after leaving the lab
Once a member of the lab has finished studying or working here, it may still be possible to use and contribute to
**Scilpy**. The user will be able to use the code, modify it and apply it to new diffusion MRI data in an academic
research environment **ONLY**. **Scilpy** cannot be used for any commercial purpose (software or imaging services).

In exchange, the user agrees that Maxime Descoteaux will be a co-author on any paper produced using **Scilpy** in the
methods section of this paper. Moreover, if the methods used were published or invented by a fellow student of the SCIL
team, this student will also be a co-author. Finally, if Jean-Christophe Houde gives support, training and assistance
on the methods, he will also be considered a co-author.

For example, if bundle registration of Garyfallidis et al. is used, Garyfallidis and Descoteaux will be co-author. Of
course, if only a small script to convert gradients is used (convert_grad-mrtrix2fsl.py), then co-authorship will not
be expected but acknowledgements would be appreciated.

[1]:http://scil.dinf.usherbrooke.ca/
[Dipy]:http://dipy.org
