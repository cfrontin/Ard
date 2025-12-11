import glob
import numpy as np
import openmdao.api as om


casefile = 'cases.sql'

def postproc_workdir(dpath,prefix,full_hist=False):
    Nprob = len(glob.glob(f'{dpath}/{prefix}*_out'))
    print(Nprob,'problems found')

    lcoe_hist = []
    x_turbine = []
    y_turbine = []
    exclusion_dist = []
    for probnum in range(Nprob):
        recfiles = glob.glob(f'{dpath}/{prefix}{probnum:02d}_out/{casefile}')
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
