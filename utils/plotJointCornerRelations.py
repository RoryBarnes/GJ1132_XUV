#!/usr/bin/env python3
"""
Corner plots of the joint hierarchical refit posteriors, for the manuscript.

For one fit variant (e.g. midlate_nosd) this renders two corner figures: the
rotation-relation block (a, b, c, d, and the scatter law's e, f) and the
activity-relation block. Splitting the twelve hyperparameters into their two
physical blocks keeps each figure legible; the cross-block correlations are
weak by construction (the relations couple only through the latent field-star
ages).

Usage: python plotJointCornerRelations.py --tag midlate_nosd
       --joint-chain jointChain_midlate_nosd.npy
       --output-directory Plot --figure-type pdf
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot  # noqa: F401  (applies the project figure style on import)

SA_ROTATION_LABELS = [r"$a_{\rm rot}$", r"$b_{\rm rot}$", r"$c_{\rm rot}$",
                      r"$d_{\rm rot}$ [d]", r"$e_{\rm rot}$", r"$f_{\rm rot}$"]
SA_ACTIVITY_LABELS = [r"$a_{\rm act}$", r"$b_{\rm act}$", r"$c_{\rm act}$",
                      r"$d_{\rm act}$", r"$e_{\rm act}$", r"$f_{\rm act}$"]


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Corner plots of the joint hierarchical refit posteriors.")
    parser.add_argument("--tag", required=True,
                        help="Fit variant tag, e.g. midlate_withsd.")
    parser.add_argument("--joint-chain", required=True,
                        help="Path to the flattened hyperparameter chain.")
    parser.add_argument("--output-directory", default="Plot",
                        help="Directory receiving the two corner figures.")
    parser.add_argument("--figure-type", default="pdf",
                        help="Figure file extension, e.g. pdf or png.")
    return parser.parse_args()


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
    args = ftParseArguments()
    os.makedirs(args.output_directory, exist_ok=True)
    daChain = np.load(args.joint_chain)
    fnRenderCorner(daChain[:, 0:6], SA_ROTATION_LABELS,
                   f"Rotation-age relation ({args.tag})",
                   os.path.join(args.output_directory,
                                f"CornerJointRotation_{args.tag}"
                                f".{args.figure_type}"))
    fnRenderCorner(daChain[:, 6:12], SA_ACTIVITY_LABELS,
                   f"Activity-age relation ({args.tag})",
                   os.path.join(args.output_directory,
                                f"CornerJointActivity_{args.tag}"
                                f".{args.figure_type}"))


if __name__ == "__main__":
    main()
