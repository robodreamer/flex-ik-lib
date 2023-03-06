from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Flexible IK Solver for Redundant Manipulators'
LONG_DESCRIPTION = \
    'This package contains a flexible inverse kinematics solver for redundant manipulators. \
    It is based on the SNS-IK algorithm, which is a multi-task prioritization algorithm \
    that can handle joint limits, joint velocity limits, and task velocity limits.'

# Setting up
setup(
       # the name must match the folder name 'flex-ik-solver'
        name="flex-ik-solver",
        version=VERSION,
        author="Andy Park",
        author_email="<andypark.purdue@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy"], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'ik', 'inverse kinematics', 'sns', 'sns-ik', 'redundant manipulators', 'robotics', 'robotics research'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
        ]
)