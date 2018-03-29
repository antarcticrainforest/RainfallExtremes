from distutils.core import setup, Extension
import numpy.distutils.misc_util, os
meta = dict(\
        description="Module to calculate radar rainfall estimates errors based on percentiles of errors",
        url = 'https://github.com/antarcticrainforest/RainfallExtremes',
        author = 'Martin Bergemann',
        author_email = 'martin.bergemann@met.fu-berlin.de',
        license = 'GPL',
        version = '1.0')

sourcef = ['_raerr.c','read_error_stats.c','calculate_pdfs.c',
          'deallocate.c','distance.c','find_percentile_error.c','lncdf.c']
source = [os.path.join('src',s) for s in sourcef]
#source = ['_raerr.c']
setup(name='radar_error',
        packages=['radar_error'],
      ext_modules=[Extension('_radar', source )],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()+['src'], **meta)
