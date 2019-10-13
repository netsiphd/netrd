import setuptools


with open('requirements.txt') as file:
    requires = [line.strip() for line in file if not line.startswith('#')]

with open('README.md') as fin:
    # read the first section of README - set between the first two '#' lines -
    # as long_description, and use the first section header as description.
    long_description = ""
    at_first_section = False
    read = iter(fin.readlines())
    for line in read:
        if at_first_section:
            break
        at_first_section = line.startswith('#')
        description = line[1:].strip()
    long_description += line
    for line in read:
        if line.startswith('#'):
            break
        long_description += line
    long_description = long_description.strip()


setuptools.setup(
    name='netrd',
    version='0.2.0',
    author='NetSI 2019 Collabathon Team',
    author_email='stefanmccabe@gmail.com',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/netsiphd/netrd',
    packages=setuptools.find_packages(),
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    extras_require={'doc': ['POT>=0.5.1']},
)
