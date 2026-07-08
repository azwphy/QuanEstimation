import os
import h5py
import numpy as np

def _deref(fl, elem):
    """Resolve JLD2 reference or return direct data as-is."""
    if isinstance(elem, np.ndarray):
        return elem
    return np.array(fl[elem])

def load_and_save(filepath, dataset_key, np_save_path, savefile,
                  item_count=0, max_episode=0, complex_view=False,
                  nested=True):
    """
    Read HDF5 file written by Julia and save as .npy.

    Access pattern is chosen by ``nested`` (orthogonal to ``complex_view``):
      - nested=True: two-level JLD2 reference (controls, measurements)
      - nested=False: direct/single-level reference (states)

    JLD2 may store data either as HDF5 object references (deref needed)
    or as raw arrays (PSO optimizer); ``_deref`` handles both transparently.

    Args:
        filepath: Source .dat file path.
        dataset_key: Dataset key in HDF5 (e.g. 'controls', 'states').
        np_save_path: Destination .npy file path (without .npy suffix).
        savefile: If True, load per-episode data; else final episode only.
        item_count: Number of items (controls/measurements). Unused for states.
        max_episode: Max episode count (not used; iteration is bounded by
                     the actual dataset length).
        complex_view: If True, view result as complex (applied after assembly).
        nested: If True, use nested-reference read (controls/measurements);
                else direct-reference read (states).
    """
    if not os.path.exists(filepath):
        return

    with h5py.File(filepath, 'r') as fl:
        dset = fl[dataset_key]

        if savefile:
            if nested:
                data = np.array([
                    [_deref(fl, fl[dset[i]][j]) for j in range(item_count)]
                    for i in range(len(dset))
                ])
            else:
                data = np.array([_deref(fl, dset[i]) for i in range(len(dset))])
        else:
            if nested:
                data = np.array([_deref(fl, dset[j]) for j in range(item_count)])
            else:
                data = np.array(dset)

        if complex_view:
            data = data.view('complex')

    np.save(np_save_path, data)
    os.remove(filepath)
