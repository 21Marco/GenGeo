import numpy as np

from utils.fluidStateFromPT import FluidStateFromPT
from utils.solver import Solver

class DownHolePumpOutput(object):
    """DownHolePumpOutput."""
    w_pump = None
    state = None

class DownHolePump(object):
    """DownHolePump."""

    def __init__(self, well, params):
        self.well = well
        self.params = params

    def solve(self, initial_state, m_dot, time_years, P_inj_surface):

        self.initial_state = initial_state
        self.P_inj_surface = P_inj_surface

        # Now pumping and second well
        d_dP_pump = 1e5
        dP_pump = 0
        dP_surface = np.nan
        dP_loops = 1
        dP_Solver = Solver()

        while np.isnan(dP_surface) or abs(dP_surface) > 100:
            try:
                if dP_pump > self.params.max_pump_dP:
                    dP_pump = self.params.max_pump_dP

                P_prod_pump_out = self.initial_state.P_Pa() + dP_pump
                T_prod_pump_out = self.initial_state.T_C()
                temp_at_pump_depth = self.well.T_e_initial + self.params.dT_dz * self.params.pump_depth

                state_in = FluidStateFromPT(P_prod_pump_out, T_prod_pump_out, self.params.working_fluid)
                well_state = self.well.solve(state_in)

                dP_surface = (well_state.state.P_Pa() - self.P_inj_surface)
                if dP_pump == self.params.max_pump_dP:
                    break

                # No pumping is needed in this system
                if dP_pump == 0 and dP_surface >= 0:
                    break

                dP_pump = dP_Solver.addDataAndEstimate(dP_pump, dP_surface)
                if np.isnan(dP_pump):
                    dP_pump = -1 * dP_surface

                # Pump can't be less than zero
                if dP_pump < 0:
                    dP_pump = 0

                # Warn against excessive loops
                if dP_loops > 10:
                    print('Warning::DownHolePump:dP_loops is large: %s'%dP_loops)
                dP_loops += 1

            except ValueError as error:
                # Only catch problems of flashing fluid
                if str(error).find('GenGeo::SemiAnalyticalWell:BelowSaturationPressure') > -1:
                    dP_pump = dP_pump + d_dP_pump
                else:
                   raise error
        # if pump pressure greater than allowable, throw error
        if dP_pump >= self.params.max_pump_dP:
            raise Exception('DownHolePump:ExceedsMaxProductionPumpPressure - '
            'Exceeds Max Pump Pressure of %.3f MPa!' %(self.params.max_pump_dP/1e6))

        self.output = DownHolePumpOutput()
        self.output.w_pump = 0
        if dP_pump > 0:
            self.output.w_pump = (self.initial_state.h_Jkg() - state_in.h_Jkg()) / self.params.eta_pump

        return well_state

    def gatherOutput(self):
        return self.output
