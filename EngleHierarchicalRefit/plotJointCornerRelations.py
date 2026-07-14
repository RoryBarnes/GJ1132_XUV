#!/usr/bin/env python3
"""
Corner plots of the joint hierarchical refit posteriors, for the manuscript.

For one fit variant (e.g. midlate_nosd) this renders two corner figures: the
rotation-relation block (a, b, c, d, and the scatter law's e, f) and the
activity-relation block. Splitting the twelve hyperparameters into their two
physical blocks keeps each figure legible; the cross-block correlations are
weak by construction (the relations couple only through the latent field-star
ages).

Usage: python plotJointCornerRelations.py <tag> [outputDirectory] [figureType]
       e.g. python plotJointCornerRelations.py midlate_nosd ../Plot pdf
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot  # noqa: F401  (applies the project figure style on import)

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

SA_ROTATION_LABELS = [r"$a_{\rm rot}$", r"$b_{\rm rot}$", r"$c_{\rm rot}$",
                      r"$d_{\rm rot}$ [d]", r"$e_{\rm rot}$", r"$f_{\rm rot}$"]
SA_ACTIVITY_LABELS = [r"$a_{\rm act}$", r"$b_{\rm act}$", r"$c_{\rm act}$",
                      r"$d_{\rm act}$", r"$e_{\rm act}$", r"$f_{\rm act}$"]


def fdaLoadHyperChain(sTag):
    """Load the flattened hyperparameter chain for one fit variant."""
    return np.load(os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy"))


def fnRenderCorner(daBlock, saLabels, sTitle, sOutputPath):
    """Render one six-parameter corner figure and save it."""
    figure = corner.corner(daBlock, labels=saLabels, show_titles=True,
                           title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
                           label_kwargs={"fontsize": 12})
    figure.suptitle(sTitle, fontsize=13)
    figure.savefig(sOutputPath, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {sOutputPath}")


def main():
    """Render the rotation and activity corner figures for one fit variant."""
    sTag = sys.argv[1]
    sOutputDirectory = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(S_DIRECTORY, "Plot")
    sFigureType = sys.argv[3] if len(sys.argv) > 3 else "pdf"
    os.makedirs(sOutputDirectory, exist_ok=True)
    daChain = fdaLoadHyperChain(sTag)
    fnRenderCorner(daChain[:, 0:6], SA_ROTATION_LABELS,
                   f"Rotation-age relation ({sTag})",
                   os.path.join(sOutputDirectory,
                                f"CornerJointRotation_{sTag}.{sFigureType}"))
    fnRenderCorner(daChain[:, 6:12], SA_ACTIVITY_LABELS,
                   f"Activity-age relation ({sTag})",
                   os.path.join(sOutputDirectory,
                                f"CornerJointActivity_{sTag}.{sFigureType}"))


if __name__ == "__main__":
    main()
