# Setup

```
conda create --name thesis python=3.12
conda activate thesis
pip install -e .
```
If you get an OpenGL error due to the robotic GUI try

    conda install -c conda-forge libgcc-ng=14 libstdcxx-ng=14 libgomp=14 libnsl libxcrypt