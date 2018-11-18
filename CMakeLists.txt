cmake_minimum_required(VERSION 3.5)
project(semantic_segmentation)
find_package(catkin REQUIRED)
catkin_metapackage()
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
