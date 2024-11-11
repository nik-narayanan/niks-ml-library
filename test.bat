@echo off

if not exist build (
    echo Creating build directory...
    mkdir build
    if %errorlevel% neq 0 (
        echo Failed to create build directory.
        exit /b %errorlevel%
    )
    
    cd build
    echo Running initial CMake configuration...
    cmake ..
    if %errorlevel% neq 0 (
        echo CMake configuration failed.
        exit /b %errorlevel%
    )
) else (
    cd build
)

echo Building the project...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)

echo Running tests...
ctest -C Release
if %errorlevel% neq 0 (
    echo Tests failed.
    exit /b %errorlevel%
)

cd ..
echo.
echo ***Running Tests Completed***
