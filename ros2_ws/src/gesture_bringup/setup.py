from glob import glob

from setuptools import setup

package_name = "gesture_bringup"

setup(
    name=package_name,
    version="3.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/models", glob("models/*.tflite") + glob("models/*.labels.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nguyen Pha Phim",
    maintainer_email="phaphim2905@gmail.com",
    description="Launch, config, model cho hệ điều khiển drone bằng cử chỉ.",
    license="MIT",
)
