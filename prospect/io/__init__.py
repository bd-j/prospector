from . import write_results
from . import read_results

__all__ = ["write_results", "read_results", "NumpyEncoder"]


class NumpyEncoder(json.JSONEncoder):
    """
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        return json.JSONEncoder.default(self, obj)
