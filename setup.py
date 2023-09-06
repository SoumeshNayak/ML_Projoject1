from setuptools import find_packages,setup
from typing import List

Hypen_e='-e .'

def get_requirements(file_path:str)-> List[str]:
    '''
    This function will retuen list of req
    '''
    requ=[]
    with open(file_path) as file_obj:
        requ=file_obj.readlines()
        requ=[req.replace('\n',"") for req in requ]
        if Hypen_e in requ:
            requ.remove(Hypen_e)
    return requ    
setup(
    name='mlops_project',
    version='0.0.1',
    author='Soumesh Nayak',
    author_email='soumeshnayak2210@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)