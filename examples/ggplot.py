from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects import pandas2ri
pandas2ri.activate()

import os
import tempfile
import uuid
from IPython.core.display import Image

grdevices = importr('grDevices')
def ggplot_notebook(gg, width = 800, height = 600):
    # adapted from: http://stackoverflow.com/a/15343186
    fn = os.path.join(tempfile.gettempdir(), '{uuid}.png'.format(uuid = uuid.uuid4()))
    grdevices.png(fn, width = width, height = height)
    gg.plot()
    grdevices.dev_off()
    return Image(filename=fn)