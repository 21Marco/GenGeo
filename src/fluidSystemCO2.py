import numpy as np

from src.parasiticPowerFractionCoolingTower import parasiticPowerFractionCoolingTower
from src.powerPlantOutput import PowerPlantEnergyOutput

from utils.fluidStateFromPT import FluidStateFromPT
from utils.fluidStateFromPh import FluidStateFromPh
from utils.solver import Solver

class FluidSystemCO2Output(object):
    """FluidSystemCO2Output."""
    pass

class FluidSystemCO2(object):
    """FluidSystemCO2 provides methods to compute the water fluid cycle."""

    def __init__(self, params):

        self.params = params

        self.injection_well = None
        self.reservoir = None
        self.production_well = None

    def solve(self):

        results = FluidSystemCO2Output()
        results.pp = PowerPlantEnergyOutput()

        # Find condensation pressure
        T_condensation = self.params.T_ambient_C + self.params.dT_approach
        P_condensation = FluidStateFromPT.getPFromTQ(T_condensation, 0, self.params.working_fluid) + 50e3
        dP_pump = 0

        dP_downhole_threshold = 1e3
        dP_downhole = np.nan
        dP_loops = 1
        dP_solver = Solver()
        while np.isnan(dP_downhole) or dP_downhole >= dP_downhole_threshold:

            # Find Injection Conditions
            P_pump_inlet = P_condensation
            P_pump_outlet = P_pump_inlet + dP_pump

            T_pump_inlet = T_condensation
            pump_inlet_state = FluidStateFromPT(P_pump_inlet, T_pump_inlet, self.params.working_fluid)

            h_pump_outlet = pump_inlet_state.h_Jkg()
            if dP_pump > 0:
                h_pump_outletS = FluidStateFromPT.getHFromPS(P_pump_outlet, pump_inlet_state.S_JK(), self.params.working_fluid)
                h_pump_outlet = pump_inlet_state.h_Jkg() + (h_pump_outletS - pump_inlet_state.h_Jkg()) / self.params.eta_pump_co2

            results.pp.w_pump = -1 * (h_pump_outlet - pump_inlet_state.h_Jkg())

            surface_injection_state = FluidStateFromPh(P_pump_outlet, h_pump_outlet, self.params.working_fluid)

            results.injection_well    = self.injection_well.solve(surface_injection_state)

            results.reservoir         = self.reservoir.solve(results.injection_well.state)

            # find downhole pressure difference (negative means
            # overpressure
            dP_downhole = self.params.P_reservoir() - results.reservoir.state.P_Pa()
            dP_pump = dP_solver.addDataAndEstimate(dP_pump, dP_downhole)
            if np.isnan(dP_pump):
                dP_pump = 0.5 * dP_downhole

            if dP_loops > 10:
                print('GenGeo::Warning::FluidSystemCO2:dP_loops is large: %s'%dP_loops)
            dP_loops += 1

            # dP_pump can't be negative
            if dP_pump < 0 and dP_downhole > 0:
                dP_pump = 0

        if results.reservoir.state.P_Pa() >= self.params.P_reservoir_max():
            raise Exception('GenGeo::FluidSystemCO2:ExceedsMaxReservoirPressure - '
                        'Exceeds Max Reservoir Pressure of %.3f MPa!'%(self.params.P_reservoir_max()/1e6))

        results.production_well  = self.production_well.solve(results.reservoir.state)

        # Calculate Turbine Power
        h_turbine_outS = FluidStateFromPT.getHFromPS(P_condensation, results.production_well.state.S_JK(), self.params.working_fluid)
        h_turbine_out = results.production_well.state.h_Jkg() - self.params.eta_turbine_co2 * (results.production_well.state.h_Jkg() - h_turbine_outS)

        results.pp.w_turbine = results.production_well.state.h_Jkg() - h_turbine_out
        if results.pp.w_turbine < 0:
            raise Exception('GenGeo::FluidSystemCO2:TurbinePowerNegative - Turbine Power is Negative')

        # heat rejection
        h_satVapor = FluidStateFromPT.getHFromPQ(P_condensation, 1,  self.params.working_fluid)
        h_condensed = FluidStateFromPT.getHFromPQ(P_condensation, 0,  self.params.working_fluid)
        if h_turbine_out > h_satVapor:
            # desuperheating needed
            results.pp.q_cooler = h_satVapor - h_turbine_out
            results.pp.q_condenser = h_condensed - h_satVapor
            T_turbine_out = FluidStateFromPT.getTFromPh(P_condensation, h_turbine_out, self.params.working_fluid)
            dT_range = T_turbine_out - T_condensation
        else:
            # no desuperheating
            results.pp.q_cooler = 0
            results.pp.q_condenser = h_condensed - h_turbine_out
            dT_range = 0

        parasiticPowerFraction = parasiticPowerFractionCoolingTower(self.params.T_ambient_C, self.params.dT_approach, dT_range, self.params.cooling_mode)
        results.pp.w_cooler = results.pp.q_cooler * parasiticPowerFraction('cooling')
        results.pp.w_condenser = results.pp.q_condenser * parasiticPowerFraction('condensing')

        results.pp.w_net = results.pp.w_turbine + results.pp.w_pump + results.pp.w_cooler + results.pp.w_condenser

        results.pp.dP_surface = results.production_well.state.P_Pa() - P_condensation

        results.pp.state_out = surface_injection_state
        self.output = results
        return results

    def gatherOutput(self):
        output = FluidSystemCO2Output()
        return self.output
