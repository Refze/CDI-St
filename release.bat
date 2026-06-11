@echo off
if "%1"=="" (
    echo Usage: release.bat 0.1.4
    exit /b 1
)
set VERSION=%1

echo Cleaning old build artifacts...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist src\cdi_stools.egg-info rmdir /s /q src\cdi_stools.egg-info
if exist src\cdi_st.egg-info rmdir /s /q src\cdi_st.egg-info

echo.
echo Current version sources:
type pyproject.toml | findstr "version"
type src\cdi_st\__init__.py | findstr "__version__"

echo.
echo Make sure both say %VERSION% above. If not, edit them now in Notepad and save.
pause

echo Building...
python -m build

echo.
echo Built files:
dir dist

echo.
echo Running twine check...
python -m twine check dist\*

echo.
echo Ready. Run this to upload:
echo   python -m twine upload dist\*