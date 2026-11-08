# First fit: Python

Every task has a class with the same parameters as the parameter files: the keywords become
arguments, in lower case. A ring parameter is one value for all the rings or one value per ring.

```python
import numpy as np
from bbarolo import FitMod3D

f = FitMod3D("ngc2403.fits", radii=np.arange(15, 600, 30), xpos=77, ypos=77, vsys=132.8,
             vrot=120, vdisp=8, inc=60, pa=123.7, z0=10, free="VROT VDISP", mask="SEARCH")
res = f.run(threads=8)     # in memory: nothing is written

res.rings                  # an astropy Table: radii, vrot, vdisp, inc, pa, ... and the errors
res.rotation_curve         # radius, VROT and its errors
res.model                  # the model cube (an astropy HDU)
res.plot_rings()           # a matplotlib Figure of the fitted parameters
res.write()                # the files of the command line, in output/ngc2403/
```

`run()` computes in memory and returns a `Result` with the outputs. `res.write()` writes the files
and the plots of the command line in the output folder (`outfolder=...` of the task, or
`res.write("folder/")`). The cube can also be an astropy HDU or a `(data, header)` tuple.

A parameter file runs from Python as from the command line, and writes its outputs:

```python
import bbarolo
out = bbarolo.run("ngc2403.par", threads=8)     # {task: Result}
bbarolo.help("3DFIT")                           # the parameters of a task
bbarolo.help("free")                            # one parameter
```

{doc}`../user-guide/python` describes the interface, and {doc}`../api/index` lists the classes of all
the tasks.
