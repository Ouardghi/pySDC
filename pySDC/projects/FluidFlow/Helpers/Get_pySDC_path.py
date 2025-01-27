import os

from pathlib import Path


def Get_pySDC_Path():
    '''
    This script returns the path to the FEniCS project directory, regardless of the location
    of the 'pySDC-master' folder in the user system and the working directory withing
    'pySDC-master' folder from which the code is called.
    '''

    # Get the working directory
    path = os.getcwd()

    # Get the index of the 'pySDC-master' in the path
    index = path.find("pySDC-master")

    # Check if "pySDC-master" exists in the path
    if index != -1:
        # Remove all characters after "pySDC-master"
        result = path[: index + len("pySDC-master")]
    else:
        # "pySDC-master" doesn't exist in the path
        index2 = path.find("pySDC")

        # Check if "pySDC-master" exists in the path
        if index2 != -1:
            # Remove all characters after "pySDC-master"
            result = path[: index2 + len("pySDC")]

    # Modify the path to the folder of the FEniCS project
    result = result + "/pySDC/projects/FluidFlow/"
    return result
