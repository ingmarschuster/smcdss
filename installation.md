## Installation ##

The Sequential Monte Carlo algorithm is written in Python 2.7 and requires several auxiliary modules that come with the project.

  * Download the project and unzip where ever you like.

  * Open the file _bin/int.sh_ and set
```
 PROJECT_FOLDER=
```
> to the root of the project folder.

  * Create a symbolic link
```
 sudo ln -s $HOME/PROJECT_FOLDER/bin/int.sh /usr/local/sbin/int
```
> where PROJECT\_FOLDER should be the root of the project folder.

## External packages ##

The project uses several external packages that need to be installed.
  * [SciPy and NumPy](http://www.scipy.org/) for scientific computing and fast array manipulations.
  * [Parallel Python](http://www.parallelpython.com/) for parallelized computing.
  * [Cython](http://cython.org) for C-extensions to speed up some loops