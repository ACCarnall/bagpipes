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

                nan_mask = cats[-1]["#ID"].isnull()
                cats[-1] = cats[-1].groupby(nan_mask).get_group(False)

                if isinstance(cats[-1].loc[0, "#ID"], float):
                    cats[-1]["#ID"] = cats[-1]["#ID"].astype(int)
                    cats[-1]["#ID"] = cats[-1]["#ID"].astype(str)

                cats[-1].index = cats[-1]["#ID"]
                break

            except ValueError:
                time.sleep(1)

    # Generate files to merge outputs into
    all_IDs = np.loadtxt("pipes/cats/" + run + "/IDs", dtype=str)

    finalcat = pd.concat(cats)
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

        for ID in all_IDs:
            obj_path = "pipes/posterior/" + run + "/" + ID
            remove = glob.glob("pipes/cats/" + run + "/" + ID + ".lock")
            remove += glob.glob(obj_path + "*.txt")
            remove += glob.glob(obj_path + "*.dat")
            remove += glob.glob(obj_path + "*.points")

            for file in remove:
                call(["rm", file])

            if os.path.exists(obj_path + ".h5"):
                os.system("touch pipes/cats/" + run + "/" + ID + ".lock")

    # Print an update about the number of fitted objects
    print("Bagpipes:", finalcat.shape[0], "out of",
          all_IDs.shape[0], "objects completed.")

    if mode == "clean":
        print("Bagpipes: Partially completed objects reset.")


def clean(run):
    """ Run compile_cat with the clean option enabled to kill running
    processes and delete progress for uncompleted objects. """
    merge(run, mode="clean")
