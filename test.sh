#!/bin/bash

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
    if [ $? -ne 0 ]; then
        echo "Failed to create build directory."
        exit $?
    fi

    cd build
    echo "Running initial CMake configuration..."
    cmake ..
    if [ $? -ne 0 ]; then
        echo "CMake configuration failed."
        exit $?
    fi
else
    cd build
fi

echo "Building the project..."
cmake --build . --config Release
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit $?
fi

echo "Running tests..."
ctest -C Release
if [ $? -ne 0 ]; then
    echo "Tests failed."
    exit $?
fi

cd ..
echo
echo ***Running Tests Completed***
