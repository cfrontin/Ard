import glob
import numpy as np
import ast
import openmdao.api as om


casefile = 'cases.sql'

def postproc_workdir(dpath,prefix,full_hist=False):
    outdirs = sorted(glob.glob(f'{dpath}/{prefix}*_out'))
    print(len(outdirs),'problems found')

    lcoe_hist = []
    x_turbine = []
    y_turbine = []
    exclusion_dist = []
    for outdir in outdirs:
        recfiles = glob.glob(f'{outdir}/{casefile}')
        assert len(recfiles) == 1
        print(recfiles[0])

        # Extract the driver cases
        cr = om.CaseReader(recfiles[0])
        cases = cr.get_cases("driver")

        # Loop through the cases and extract iteration number and objective value
        lcoe = []
        xturb, yturb = [], []

        if full_hist:
            cases_to_process = cases
        else:
            cases_to_process = [cases[-1]]

        for case in cases_to_process:
            lcoe.append(case.get_val('financese.lcoe', units='USD/MW/h'))
            xturb.append(case.get_val('x_turbines', units='m'))
            yturb.append(case.get_val('y_turbines', units='m'))
        lcoe_hist.append(np.array(lcoe).squeeze())
        x_turbine.append(np.array(xturb))
        y_turbine.append(np.array(yturb))

        # Extract turbine layout info -- no need to get full history of exclusion dist right now
        try:
            exclusion_dist.append(np.array(cases[-1].get_val("exclusions.exclusion_distances", units="km")))
        except KeyError:
            pass

    return lcoe_hist, x_turbine, y_turbine, exclusion_dist


def scrape_log(log):
    lcoe_vals = []
    xturb_vals = []
    yturb_vals = []
    dvstr = ''
    readdv = False
    with open(log, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("'LCOE_val'"):
                lcoe_vals.append(float(line.rstrip(',').split()[-1]))

                # parse turb locs (can't handle numpy array)
                dvstr = dvstr.replace('array(','')
                dvstr = dvstr.replace(')','')
                dv = ast.literal_eval(dvstr)

                xturb_vals.append(np.array(dv['x_turbines']))
                yturb_vals.append(np.array(dv['y_turbines']))

            elif line == 'Design Vars':
                assert readdv == False
                readdv = True
                dvstr = ''
            elif line == '' and readdv:
                readdv = False
            elif readdv:
                dvstr += line

    return lcoe_vals, xturb_vals, yturb_vals


def normalize_layout(x, y):
    """
    Normalize a layout by sorting points lexicographically.
    This makes layouts with the same points but different orders comparable.
    """
    points = np.column_stack([x, y])
    # Sort by x, then by y
    sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))]
    return sorted_points

def are_layouts_equal(x1, y1, x2, y2, rtol=1e-5, atol=1e-8):
    """
    Check if two layouts are equal within tolerance.
    """
    if len(x1) != len(x2):
        return False

    layout1 = normalize_layout(x1, y1)
    layout2 = normalize_layout(x2, y2)

    return np.allclose(layout1, layout2, rtol=rtol, atol=atol)

def find_unique_layouts(x_list, y_list):
    """
    Find unique layouts from lists of x and y coordinate arrays.
    Returns indices of unique layouts.
    """
    n_layouts = len(x_list)
    is_unique = np.ones(n_layouts, dtype=bool)

    for i in range(n_layouts):
        if not is_unique[i]:
            continue
        for j in range(i + 1, n_layouts):
            if is_unique[j] and are_layouts_equal(x_list[i], y_list[i],
                                                   x_list[j], y_list[j]):
                is_unique[j] = False

    return np.where(is_unique)[0]
