from __future__ import print_function, division, absolute_import

import numpy as np
import glob
import pandas as pd
import os

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
                cats.append(pd.read_table(file, delimiter="\t",
                                          names=header, skiprows=1))

                if isinstance(cats[-1].loc[0, "#ID"], float):
                    cats[-1].loc[:, "#ID"] = cats[-1].loc[:, "#ID"].astype(int)

                cats[-1].index = cats[-1]["#ID"].astype(str)
                break

            except ValueError:
                time.sleep(1)

    # Generate files to merge outputs into
    all_IDs = np.loadtxt("pipes/cats/" + run + "/all_IDs", dtype=str)

    finalcat = pd.DataFrame(np.zeros((all_IDs.shape[0], cats[0].shape[1])),
                            columns=header, index=all_IDs)

    finalcat.loc[:, "#ID"] = all_IDs

    # Merge outputs into final catalogue
    for ind in finalcat.index:
        for cat in cats:
            if ind in cat.index:
                finalcat.loc[ind, :] = cat.loc[ind, :]
                break

        else:
            finalcat.loc[ind, "#ID"] = np.nan

    finalcat = finalcat.groupby(finalcat["#ID"].isnull()).get_group(False)
    finalcat.to_csv("pipes/cats/" + run + ".cat", sep="\t", index=False)

    # If mode is clean, remove all of the separate input catalogues
    if mode == "clean":
        for file in files:
            call(["rm",  file])

        # Save the final cat as a merged input catalogue
        finalcat.to_csv("pipes/cats/" + run + "/" + run + ".txt_clean",
                        sep="\t", index=False)

        # Get list of lock files
        len_start = len("pipes/cats/" + run + "/")
        lock_files = glob.glob("pipes/cats/" + run + "/*.lock")

        # Delete all output files and lock files for these objects
        for lock_file in lock_files:
            ID = lock_file[len_start:-5]

            if ID not in finalcat.loc[:, "#ID"]:

                remove = [lock_file]
                remove += glob.glob("pipes/posterior/" + run + "/" + ID + "*")

                for file in remove:
                    call(["rm", file])

    # Print what has been done
    print("Bagpipes:", finalcat.shape[0], "out of",
          all_IDs.shape[0], "objects completed.")

    if mode == "clean":
        print("Bagpipes: Partially completed objects reset.")


def clean(run):
    """ Run compile_cat with the clean option enabled to kill running
    processes and delete progress for uncompleted objects. """
    merge(run, mode="clean")
