#!/usr/bin/env python3
"""
Simplified script to recreate Figure 10 with your own fitted parameters.

Just replace the parameters below with your MCMC results and run!
"""

import numpy as np
import matplotlib.pyplot as plt
import vplot


def ftOldConstants():
    a1 = -0.06596571
    a2 =  0.77855978
    a3 = -1.0475149
    b1 = 1.91734981
    b2 = -24.57936264
    b3 = 33.65312658

    return a1, a2, a3, b1, b2, b3


def fnPlotFFD(daLogAge, daParams, daLogEnergies, daAgesMyr, dMass,
              sDescription, sLinestyle):
    """Plot FFD curves at multiple ages for one parameter set."""
    listColors = [vplot.colors.pale_blue, vplot.colors.purple,
                  vplot.colors.orange, vplot.colors.red,
                  vplot.colors.dark_blue, 'k']
    a1, a2, a3, b1, b2, b3 = daParams
    for iAge in range(len(daLogAge)):
        dSlope = a1 * daLogAge[iAge] + a2 * dMass + a3
        dIntercept = b1 * daLogAge[iAge] + b2 * dMass + b3
        daFfd = 10**(dSlope * daLogEnergies + dIntercept)
        daEnergies = 10**daLogEnergies
        sLabel = repr(daAgesMyr[iAge]) + ' Myr (' + sDescription + ')'
        plt.plot(daEnergies, daFfd, color=listColors[iAge],
                 linestyle=sLinestyle, label=sLabel)


def fnPlot(daParams, dMass=0.5, sFilename='ffd_comp.png'):
    """Create Figure 10 comparing old and new fitted parameters."""
    daLogEnergies = np.linspace(33, 36, 100)
    listAgesMyr = [10, 100, 1000, 10000]
    daLogAge = np.log10(listAgesMyr)

    plt.figure(figsize=(6.5, 6))

    tOldParams = ftOldConstants()

    fnPlotFFD(daLogAge, tOldParams, daLogEnergies, listAgesMyr,
              dMass, 'old', 'dashed')
    fnPlotFFD(daLogAge, daParams, daLogEnergies, listAgesMyr,
              dMass, 'new', 'solid')

    plt.xlabel('log Flare Energy (erg)', fontsize=18)
    plt.ylabel('Cumulative Flare Freq (#/day)', fontsize=18)
    plt.xlim(8e32, 1.05e36)
    plt.ylim(3e-5, 3e-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([1e33, 1e34, 1e35, 1e36], ['33', '34', '35', '36'],
               fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', fontsize=14)
    plt.annotate(f'M = 0.5 M$_\\odot$', [2e33, 2.5e-3], fontsize=14)
    plt.tight_layout()
    plt.savefig(sFilename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {sFilename}")


if __name__ == "__main__":
    try:
        daSamples = np.load('flare_mcmc_samples.npy')

        listFittedParams = [np.median(daSamples[:, i]) for i in range(6)]
        listUncertainties = [np.std(daSamples[:, i]) for i in range(6)]

        print("Parameter values (median +/- std):")
        listParamNames = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']
        for sName, dValue, dError in zip(listParamNames, listFittedParams,
                                         listUncertainties):
            print(f"  {sName}: {dValue:.4f} +/- {dError:.4f}")

    except (FileNotFoundError, ImportError):
        print("Could not load MCMC results!")
        exit()

    import sys
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else "ffd_comp.pdf"
    fnPlot(listFittedParams, dMass=0.5, sFilename=sOutputPath)
