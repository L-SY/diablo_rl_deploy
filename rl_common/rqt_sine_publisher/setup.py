from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['rqt_sine_publisher'],
    package_dir={'': 'src'},
    scripts=[],
    requires=['std_msgs', 'rospy', 'rqt_gui', 'rqt_gui_py']
)

setup(**d)