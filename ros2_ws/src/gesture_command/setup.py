from setuptools import setup

package_name = "gesture_command"

setup(
    name=package_name,
    version="3.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nguyen Pha Phim",
    maintainer_email="phaphim2905@gmail.com",
    description="Cử chỉ -> lệnh drone an toàn qua MAVROS. Không bao giờ arm.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "command_node = gesture_command.command_node:main",
        ],
    },
)
