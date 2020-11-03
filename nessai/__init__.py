from importlib.metadata import version, PackageNotFoundError
import logging

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = ['flowsampler'
           'nestedsampler'
           'proposal',
           'model'
           'flowmodel',
           'flows',
           'utils']
