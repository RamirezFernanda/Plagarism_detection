from setuptools import setup, find_packages

setup(
    name="plagiarism_detection",
    version="1.0.0",
    description="The project presented below seeks to detect plagiarism through the similarity of cosine and n-grams in the comparison of files with possible plagiarism and their corresponding original documents",
    url="https://github.com/RamirezFernanda/Plagarism_detection",
    author="Maria Fernanda Ramirez Barragan,Melissa Garduño Ruiz",
    license="MIT",
    classifiers=['Development Status :: 2 - Beta', 'Intended Audience :: Developers', 'Topic :: Plagiarism Detection', 'Programming Language :: Python :: 3.10'],
    keywords="plagiarism detection program",
    packages=find_packages(include=['AUC', 'Detection_files', 'Results_images', 'AUC.*', 'Detection_files.*', 'Results_images.*']),
    python_requires='>=3.8'
)