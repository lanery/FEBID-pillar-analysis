from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Ellipse
from skimage import measure, color
from tifffile import TiffFile

from image_tracing import boundary_tracing


def get_outline(pillar):
    """Get pillar outline"""
    # Trace segment boundary
    coords = boundary_tracing(pillar)[:, ::-1]
    # Funky numpy groupby via https://stackoverflow.com/a/43094244/5285918
    # to get the x coordinates corresponding to each unique y coordinate
    coords = coords[coords[:, 1].argsort()]  # sort by y first
    grps_x = np.split(coords[:, 0], np.unique(coords[:, 1], return_index=True)[1][1:])
    # Get min/max x value for each y value
    x_ = np.array([(grp.min(), grp.max()) for grp in grps_x])
    y_ = np.unique(coords[:, 1])[:, np.newaxis]
    # Merge min/max x with y and call this the outline
    outline = np.hstack([y_, x_])
    return outline


def measure_volume(
    pillar,
    pixelsize,
    theta,
    method="pillar"
):
    """Measure geometric properties of a pillar 

    Parameters
    ----------
    pillar : `skimage.measure._regionprops.RegionProperties`
        Pillar segment as skimage region object.
    pixelsize : float
        Pixelsize (nm/px).
    theta : float
        Tilt angle (rad).
    method : str (optional)
        Method for measuring volume of the pillar. Options are
        1. "cone"             |  V = pi r^2 h/3
        2. "cylinder"         |  V = pi r^2 h
        3. "pillar" (default) |  V = ∫ A dz

    Returns
    -------
    width : float
        Measured width
    height : float
        Measured height
    volume : float
        Measured volume
    sigma_h : float
        Estimated error in height measurement
    sigma_v : float
        Estimated error in volume measurement
    """

    # Correct for tilt projection (Mike certified ;)
    psx = pixelsize
    psy = pixelsize * np.cos(theta)
    psz = pixelsize / np.sin(theta)

    # Get outline (y, x_min, x_max) coordinates
    outline = get_outline(pillar)

    # Get pillar bounding box (y dimensions only)
    y1, _, y2, _ = pillar.bbox

    # Overly complicated way of calculating the index
    # for which the pillar diameter is largest
    i = outline.shape[0] - np.diff(outline[::-1, 1:]).argmax() - 1
    # Get min/max x here
    x1 = outline[i, 1]
    x2 = outline[i, 2]
    # Get (x, y) coordinates of base
    xc = (x1 + x2) / 2   # x center of pillar base
    rx = (x2 - x1) / 2   # x radius of pillar base
    ry = rx * (psy/psx)  # y radius of pillar base
    yb = y2 - ry         # y center of pillar base
    # (re)set i to be index corresponding to pillar base
    i = np.argwhere(outline[:, 0] == int(yb)).item()

    # Convert to physical units
    width = (x2 - x1) * psx
    height = np.abs(y1 - yb) * psz

    # Assume 1px error in both width and height
    sigma_w = psx
    sigma_h = psz

    # Approximate volume as a cone
    if method.lower().startswith("cone"):
        volume = np.pi/3 * (width/2)**2 * height
        sigma_v = np.sqrt(((np.pi*height/3 * width/2)**2 * (sigma_w/2)**2) +\
                          ((np.pi*width/12)**2 * sigma_h**2))

    # Approximate volume as a cylinder
    elif method.lower().startswith("cylinder"):
        volume = np.pi * (width/2)**2 * height
        sigma_v = np.sqrt(((np.pi*height * width/2)**2 * (sigma_w/2)**2) +\
                          ((np.pi*width/4)**2 * sigma_h**2))

    # Calculate volume by integrating with "pixel-perfect" precision
    elif method.lower().startswith("pillar"):

        # Compute V = ∫ A dz (where dz is projected from dy)
        # | idea is to first calculate the diameter of the pillar for each y value
        # | then integrate along z axis from base to tip
        # | also have to convert to physical units
        r = psx * np.diff(outline[:i+1, 1:]).squeeze()[:-1] / 2
        A = np.pi * r**2  # area of each disk in the pillar
        dz = psz  # assume pillar is continuous such that no gap in y coordinates (spacing = 1 px)
        volume = np.sum(A * dz)
        sigma_v = np.sqrt(((np.pi*height * width/2)**2 * psx**2) +\
                          ((np.pi*width/4)**2 * psz**2))

    # Whachu tryna do?
    else:
        raise ValueError(f"Method, `{method}`, not recognized.")

    return (x1, x2), (y1, y2), (xc, yb), width, height, volume, sigma_h, sigma_v


def sort_pillars(pillars):
    """Sort pillars into grid"""
    # get pillar centroids
    points = np.array([p.centroid[::-1] for p in pillars])
    points[:, 1] *= -1  # invert y

    # uses a priori knowledge that one image has a whole column missing
    nrows = 3
    ncols = 3 if len(points) > 6 else 2

    # determine width, height of each cell in grid (add a bit of padding)
    w_cell = 20 + (points[:, 0].max() - points[:, 0].min()) / ncols
    h_cell = 20 + (points[:, 1].max() - points[:, 1].min()) / nrows

    # loop through centroids
    grid = {}
    for i, point in enumerate(points):
        x, y = point
        row = int((y - points[:, 1].min()) // h_cell)
        col = int((x - points[:, 0].min()) // w_cell)
        col = col + 1 if ncols == 2 else col  # I hate this
        grid[(col, row)] = pillars[i]

    # Sort grid
    # (0, 2) --> (1, 2) --> (2, 2)
    # ^ ------------------------ v
    # (0, 1) --> (1, 1) --> (2, 1)
    # ^ ------------------------ v
    # (0, 0) --> (1, 0) --> (2, 0)
    grid_sorted = {k: grid[k] for k in sorted(grid, key=lambda k: (k[1], k[0]))}

    return grid_sorted


def plot_pillars(
    image,
    mask,
    pixelsize,
    theta,
    method="cone",
    area_min=1000,
    title=None,
    filename="",
    dpi=300
):
    """Overlay pillar measurements on SEM image"""

    # Process masks
    labels = measure.label(mask > 65535/2)
    overlay = color.label2rgb(
        labels,
        image=image,
        colors=[(1, 0.7, 0)],
        bg_label=0,
    )
    # Collect segments from labelled image
    segments = measure.regionprops(labels)
    # Assume 9 largest segments are pillars
    pillars = sorted(
        segments,
        key=lambda x: x.area,
        reverse=True
    )[:9]
    # Filter out teeny tiny pillars
    pillars = [p for p in pillars if p.area > area_min]
    # Sort pillars into grid
    pillars_grid = sort_pillars(pillars)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(overlay)

    # Loop through pillars
    for (cc, rr), pillar in pillars_grid.items():

        # Measure pillar
        (x1, x2), (y1, y2), (xc, yb), width, height, volume, sigma_h, sigma_v = measure_volume(
            pillar=pillar,
            pixelsize=pixelsize,
            theta=theta,
            method=method
        )

        # Trace segment boundary
        coords = boundary_tracing(pillar)[:, ::-1]

        # Volume annotation patch kwargs
        patch_kwargs = {
            "fill": False,
            "alpha": 0.7,
            "edgecolor": "#8C09B3",
            "ls": "--"
        }

        # Pillar geometry patches
        top = Ellipse(
            xy=(xc, y1),
            width=(x2 - x1),
            height=(x2 - x1) * np.sin(theta),
            **patch_kwargs
        )
        bot = Ellipse(
            xy=(xc, yb),
            width=(x2 - x1),
            height=(x2 - x1) * np.sin(theta),
            **patch_kwargs
        )
        rect = Rectangle(
            xy=(x1, y1),
            width=(x2 - x1),
            height=(yb - y1),
            **patch_kwargs
        )
        triangle = Polygon(
            xy=[[x1, yb],
                [xc, y1],
                [x2, yb]],
            **patch_kwargs
        )

        # Cone volume boundary
        if method.lower().startswith("cone"):
            [ax.add_patch(p) for p in [triangle, bot]]

        # Cylinder volume boundary
        elif method.lower().startswith("cylinder"):
            [ax.add_patch(p) for p in [rect, top, bot]]

        # Pillar outline
        else:
            ax.plot(
                coords[:, 0], coords[:, 1],
                color=patch_kwargs.get("edgecolor"),
                ls=patch_kwargs.get("ls"),
                alpha=patch_kwargs.get("alpha")
            )
            [ax.add_patch(p) for p in [bot]]

        # Plot annotations
        # ----------------
        fontdict={
            "size": 7,
            "color": "white",
            "family": "monospace"
        }
        arrowprops = {
            "arrowstyle": "|-|, widthA=0.2, widthB=0.2",
            "edgecolor": "white",
        }

        # Get pillar number from grid position
        mapping = {v: k+1 for k, v in enumerate(product(range(3), range(3)))}
        p = mapping[(rr, cc)]

        # Volume
        ax.text(
            x=(x2 + coords[:, 0].max())/2,
            y=(y1 + yb)/2,
            s=f"({cc}, {rr}): {p}\n{volume:.1e}nm^3",
            va="center",
            fontdict=fontdict
        )

        # Width
        ax.text(
            x=xc,
            y=y2 + 25,
            s=f"{width:.0f}nm",
            va="top",
            ha="center",
            fontdict=fontdict
        )
        ax.annotate("",  # width bar
            xy=(x1, y2+10),
            xytext=(x2, y2+10),
            arrowprops=arrowprops
        )

        # Height
        ax.text(
            x=x1 - 20,
            y=(y1 + yb)/2,
            s=f"{height:.0f}nm",
            va="top",
            ha="right",
            fontdict=fontdict
        )
        ax.annotate("",  # height bar
            xy=(x1-10, yb),
            xytext=(x1-10, y1),
            arrowprops=arrowprops
        )

        # Aesthetics
        ax.axis('off')
        ax.set_title(title, y=0.92, color="white")

        # Optional savefig
        if filename:
            plt.savefig(
                filename,
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi
            )

    return pillars_grid
