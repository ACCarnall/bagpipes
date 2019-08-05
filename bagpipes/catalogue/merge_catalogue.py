from __future__ import print_function, division, absolute_import

import numpy as np
import glob
import pandas as pd
import os
import time

from subprocess import call


def merge(run, mode="merge"):
    """ Compile all the sub-catalogues into one output catalogue,
    optionally stop all running processes and delete incomplete object
    posteriors with the "clean" mode. """

    if mode == "clean":
        call(["touch", "pipes/cats/" + run + "/kill"])

    cats = []

    # Generate lists of files to merge
    files = glob.glob("pipes/cats/" + run + "/" + run + ".txt*")

    header = " ".join((open(files[0]).readline()[:-1]).split("\t")).split()

    # Load up files
    for file in files:
        while True:
            try:
                cats.append(pd.read_csv(file, delimiter="\t",
                                          names=header, skiprows=1))

                if isinstance(cats[-1].loc[0, "#ID"], float):
                    m = np.invert(np.isnan(cats[-1].loc[:, "#ID"]))
                    cats[-1].loc[m, "#ID"] = cats[-1].loc[m, "#ID"].astype(int)
                    cats[-1].loc[m, "#ID"] = cats[-1].loc[m, "#ID"].astype(str)

                cats[-1].index = cats[-1]["#ID"]
                break

            except ValueError:
                time.sleep(1)

    # Generate files to merge outputs into
    all_IDs = np.loadtxt("pipes/cats/" + run + "/IDs", dtype=str)

    finalcat = pd.concat(cats)
    finalcat.loc[:, "#ID"] = finalcat.loc[:, "#ID"].astype(str)
    finalcat = finalcat.groupby(finalcat["#ID"].isnull()).get_group(False)
    finalcat = finalcat.drop_duplicates(subset="#ID")
    finalcat.to_csv("pipes/cats/" + run + ".cat", sep="\t", index=False)

    # If mode is clean, remove all of the separate input catalogues
    if mode == "clean":

        # Delete separate input catalogues
        for file in files:
            call(["rm",  file])

        # Save the final cat as a merged input catalogue
        finalcat.to_csv("pipes/cats/" + run + "/" + run + ".txt_clean",
                        sep="\t", index=False)

        # Get list of lock files
        len_start = len("pipes/cats/" + run + "/")
        lock_files = glob.glob("pipes/cats/" + run + "/*.lock")

        # Delete all output files and lock files for unfinished objects
        for lock_file in lock_files:
            ID = lock_file[len_start:-5]
            obj_path = "pipes/posterior/" + run + "/" + ID

            if not np.isin(ID, finalcat.loc[:, "#ID"]):
                remove = [lock_file]
                remove += glob.glob(obj_path + "*.txt")
                remove += glob.glob(obj_path + "*.dat")
                remove += glob.glob(obj_path + "*.points")

                for file in remove:
                    call(["rm", file])

    # Print an update about the number of fitted objects
    print("Bagpipes:", finalcat.shape[0], "out of",
          all_IDs.shape[0], "objects completed.")

    if mode == "clean":
        print("Bagpipes: Partially completed objects reset.")


def clean(run):
    """ Run compile_cat with the clean option enabled to kill running
    processes and delete progress for uncompleted objects. """
    merge(run, mode="clean")
