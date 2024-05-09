# Licensed under LGPL 2.1, please see LICENSE for details
# https://www.gnu.org/licenses/lgpl-2.1.html
#
# The work on this project has been performed at the GEG Group at ETH Zurich:
# --> https://geg.ethz.ch
#
# The initial version of this file has been implemented by:
#
#     Philipp Schaedle (https://github.com/philippschaedle)
#     Benjamin M. Adams
#
# Further changes are done by:
#

############################
import os, math
import numpy as np

from src.coolingCondensingTower import CoolingCondensingTower
from src.powerPlantOutput import PowerPlantOutput

from utils.constantsAndPaths import getTboilOptimum
from utils.fluidState import FluidState
from utils.maxSubcritORCBoilTemp import maxSubcritORCBoilTemp
from models.simulationParameters import SimulationParameters


class ORCCycleTboil(object):
    """ ORCCycleTboil.
    Heat/power is output as specific heat and specific work. To find the
    actual power, multiply by the flowrate of geofluid through the system.
    """

    def __init__(self, params=None, **kwargs):
        self.params = params
        if self.params == None:
            self.params = SimulationParameters(**kwargs)
        self.data = getTboilOptimum()
        self.orc_fluid = self.params.orc_fluid
        self.T_boil_max = maxSubcritORCBoilTemp(self.orc_fluid)

    def solve(self, initialState, T_boil_C=False, dT_pinch=False):

        T_in_C = initialState.T_C

        if not T_boil_C:
            T_boil_C = np.interp(T_in_C, self.data[self.params.opt_mode][self.params.orc_fluid][:, 0],
                                 self.data[self.params.opt_mode][self.params.orc_fluid][:, 1])

        if not dT_pinch:
            dT_pinch = np.interp(T_in_C, self.data[self.params.opt_mode][self.params.orc_fluid][:, 0],
                                 self.data[self.params.opt_mode][self.params.orc_fluid][:, 2])

        # run some checks  if T_in_C and T_boil_C are valid
        if np.isnan(T_in_C):
            raise Exception('GenGeo::ORCCycleTboil:T_in_NaN - ORC input temperature is NaN!')

        if np.isnan(T_boil_C):
            raise Exception('GenGeo::ORCCycleTboil:T_boil_NaN - ORC boil temperature is NaN!')

        if T_boil_C > FluidState.getTcrit(self.params.orc_fluid):
            raise Exception('GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature above critical point')

        if dT_pinch <= 0:
            raise Exception('GenGeo::ORCCycleTboil:dT_pinch_Negative - dT_pinch is negative!')

        if T_in_C < T_boil_C + dT_pinch:
            raise Exception(
                'GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature of %s is greater than input temp of %s less pinch dT of %s.' % (
                T_boil_C, T_in_C, dT_pinch))

        # only refresh T_boil_max if orc_fluid has changed from initial
        if self.params.orc_fluid != self.orc_fluid:
            self.T_boil_max = maxSubcritORCBoilTemp(self.params.orc_fluid)
            self.orc_fluid = self.params.orc_fluid
        if T_boil_C > self.T_boil_max:
            raise Exception(
                'GenGeo::ORCCycleTboil:Tboil_Too_Large - Boiling temperature of %s is greater than maximum allowed of %s.' % (
                T_boil_C, self.T_boil_max))

        T_condense_C = self.params.T_ambient_C + self.params.dT_approach

        if self.params.orc_Saturated_SH:   #Saturated ORC Cycle, without SH
            if self.params.orc_Rec:  #without Recuperator
                state = [None] * 6

                # State 1 (Condenser -> Pump)
                # saturated liquid
                state[0] = FluidState.getStateFromTQ(T_condense_C, 0, self.params.orc_fluid)
                # p_1 = state[0].P_Pa

                # State 4 (Boiler -> Turbine)
                # saturated vapor
                state[3] = FluidState.getStateFromTQ(self.params.T_boil_C, 1, self.params.orc_fluid)

                # Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                # Condenser
                if self.params.dp_dT_loss[8] > 0:
                    p_6 = state[0].P_Pa / (1 - self.params.dp_dT_loss[8])  # dp for the condenser
                else:
                    T_6 = state[0].T_C + self.params.dp_dT_loss[8]
                    p_6 = FluidState.getStateFromTQ(T_6, 1, self.params.orc_fluid).P_Pa
                    # p_6 = dp_loss_ass_cond + state[0].P_Pa

                # Desuperheater
                if self.params.dp_dT_loss[7] > 0:
                    p_5 = p_6 / (1 - self.params.dp_dT_loss[7])  # dp for the desuperheater
                else:
                    p_5 = self.params.dp_dT_loss[7] + p_6

                # Evaporator
                if self.params.dp_dT_loss[3] > 0:
                    p_3 = state[3].P_Pa / (1 - self.params.dp_dT_loss[3])  # dp for the boiler
                else:
                    T_3 = state[3].T_C + self.params.dp_dT_loss[3]  # - dT_sc
                    p_3 = FluidState.getStateFromTQ(T_3, 0, self.params.orc_fluid).P_Pa

                # Economizer
                if self.params.dp_dT_loss[2] > 0:
                    p_2 = p_3 / (1 - self.params.dp_dT_loss[2])  # dp for the pre-heater
                else:
                    p_2 = self.params.dp_dT_loss[2] + p_3

                # Compute the TDN points
                # State 6 (Desuperheater -> Condenser)
                # saturated vapor
                state[5] = FluidState.getStateFromPQ(p_6, 1, self.params.orc_fluid)

                # State 5 (Turbine -> Desuperheater)
                h_5s = FluidState.getStateFromPS(p_5, state[3].s_JK, self.params.orc_fluid).h_Jkg
                h_5 = state[3].h_Jkg - self.params.eta_turbine_orc * (state[3].h_Jkg - h_5s)
                state[4] = FluidState.getStateFromPh(p_5, h_5, self.params.orc_fluid)

                # State 3 (Preheater -> Boiler)
                # saturated liquid
                state[2] = FluidState.getStateFromPQ(p_3, 0, self.params.orc_fluid)

                # State 2 (Pump -> Preheater)
                h_2s = FluidState.getStateFromPS(p_2, state[0].s_JK, self.params.orc_fluid).h_Jkg
                h_2 = state[0].h_Jkg - ((state[0].h_Jkg - h_2s) / self.params.eta_pump_orc)
                state[1] = FluidState.getStateFromPh(p_2, h_2, self.params.orc_fluid)

                # State 3*
                # T_3* = T_3 - dT_sc
                # state[3*] = FluidState.getStateFromPT(p_3, T_3*, self.params.orc_fluid)

                results = PowerPlantOutput()

                # #Calculate orc heat/work
                w_pump_orc = state[0].h_Jkg - state[1].h_Jkg
                q_preheater_orc = -1 * (state[1].h_Jkg - state[2].h_Jkg)
                q_boiler_orc = -1 * (state[2].h_Jkg - state[3].h_Jkg)
                w_turbine_orc = state[3].h_Jkg - state[4].h_Jkg
                q_desuperheater_orc = -1 * (state[4].h_Jkg - state[5].h_Jkg)
                q_condenser_orc = -1 * (state[5].h_Jkg - state[0].h_Jkg)

                results.dP_pump_orc = state[1].P_Pa - state[0].P_Pa
                results.P_boil = state[2].P_Pa

            else:  #with the Recuperator

                # create empty list to compute cycle of 6 states
                state = [None] * 8

                # State 1 (Condenser -> Pump)
                # saturated liquid
                state[0] = FluidState.getStateFromTQ(T_condense_C, 0, self.params.orc_fluid)
                # p_1 = state[0].P_Pa

                # State 5 (Boiler -> Turbine)
                # saturated vapor
                state[4] = FluidState.getStateFromTQ(self.params.T_boil_C, 1, self.params.orc_fluid)

                # Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                # Condenser
                if self.params.dp_dT_loss[8] > 0:
                    p_8 = state[0].P_Pa / (1 - self.params.dp_dT_loss[8])  # dp for the condenser
                else:
                    T_8 = state[0].T_C + self.params.dp_dT_loss[8]
                    p_8 = FluidState.getStateFromTQ(T_8, 1, self.params.orc_fluid).P_Pa

                # Desuperheater
                if self.params.dp_dT_loss[7] > 0:
                    p_7 = p_8 / (1 - self.params.dp_dT_loss[7])  # dp for the desuperheater
                else:
                    p_7 = self.params.dp_dT_loss[7] + p_8

                # Boiler
                if self.params.dp_dT_loss[3] > 0:
                    p_4 = state[4].P_Pa / (1 - self.params.dp_dT_loss[3])  # dp for the boiler
                else:
                    T_4 = state[4].T_C + self.params.dp_dT_loss[3]  # - dT_sc
                    p_4 = FluidState.getStateFromTQ(T_4, 0, self.params.orc_fluid).P_Pa

                # Preheater
                if self.params.dp_dT_loss[2] > 0:
                    p_3 = p_4 / (1 - self.params.dp_dT_loss[2])  # dp for the pre-heater
                else:
                    p_3 = self.params.dp_dT_loss[2] + p_4

                # Recuperator hot side
                if self.params.dp_dT_loss[6] > 0:
                    p_6 = p_7 / (1 - self.params.dp_dT_loss[6])  # dp for the recuperator hot side
                else:
                    p_6 = self.params.dp_dT_loss[6] + p_7

                # Recuperator cold side
                if self.params.dp_dT_loss[1] > 0:
                    p_2 = p_3 * (1 - self.params.dp_dT_loss[1])  # dp for the recuperator cold side
                else:
                    p_2 = p_3 - self.params.dp_dT_loss[1]

                # compute the TDN points
                # State 8 (Desuperheater -> Condenser)
                # saturated vapor
                state[7] = FluidState.getStateFromPQ(p_8, 1, self.params.orc_fluid)

                # State 6 (Turbine -> Recuperator hpt side)
                h_6s = FluidState.getStateFromPS(p_6, state[4].s_JK, self.params.orc_fluid).h_Jkg
                h_6 = state[4].h_Jkg - self.params.eta_turbine_orc * (state[4].h_Jkg - h_6s)
                state[5] = FluidState.getStateFromPh(p_6, h_6, self.params.orc_fluid)

                # State 4 (Preheater -> Boiler)
                # saturated liquid
                state[3] = FluidState.getStateFromPQ(p_4, 0, self.params.orc_fluid)

                # State 2 (Pump -> Recuperator cold side)
                h_2s = FluidState.getStateFromPS(p_2, state[0].s_JK, self.params.orc_fluid).h_Jkg
                h_2 = state[0].h_Jkg - ((state[0].h_Jkg - h_2s) / self.params.eta_pump_orc)
                state[1] = FluidState.getStateFromPh(p_2, h_2, self.params.orc_fluid)

                # State 3 (Recuperator cold side -> Preheater)
                T_3 = state[1].T_C + self.params.dT_pp_rec
                state[2] = FluidState.getStateFromPT(p_3, T_3, self.params.orc_fluid)

                # State 7 (Recuperator hot -> Desuperheater)
                h_7 = h_6 - state[2].h_Jkg + h_2
                state[6] = FluidState.getStateFromPh(p_7, h_7, self.params.orc_fluid)

                results = PowerPlantOutput()

                # #Calculate orc heat/work
                w_pump_orc = state[0].h_Jkg - state[1].h_Jkg
                q_recuperator_orc = -1 * (state[1].h_Jkg - state[2].h_Jkg)
                q_preheater_orc = -1 * (state[2].h_Jkg - state[3].h_Jkg)
                q_boiler_orc = -1 * (state[3].h_Jkg - state[4].h_Jkg)
                w_turbine_orc = state[4].h_Jkg - state[5].h_Jkg
                q_desuperheater_orc = -1 * (state[6].h_Jkg - state[7].h_Jkg)
                q_condenser_orc = -1 * (state[7].h_Jkg - state[0].h_Jkg)

                results.dP_pump_orc = state[1].P_Pa - state[0].P_Pa
                results.P_boil = state[3].P_Pa

        else:  # ORC Cycle with SH
            if self.params.orc_Rec:  #without Recuperator

                # create empty list to compute cycle of 6 states
                state = [None] * 7

                # State 1 (Condenser -> Pump)
                # saturated liquid
                state[0] = FluidState.getStateFromTQ(T_condense_C, 0, self.params.orc_fluid)
                # p_1 = state[0].P_Pa

                # State 4 (Boiler -> Superheater)
                # saturated vapor
                state[3] = FluidState.getStateFromTQ(self.params.T_boil_C, 1, self.params.orc_fluid)

                # Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                # Condenser
                if self.params.dp_dT_loss[8] > 0:
                    p_7 = state[0].P_Pa / (1 - self.params.dp_dT_loss[8])  # dp for the condenser
                else:
                    T_7 = state[0].T_C + self.params.dp_dT_loss[8]
                    p_7 = FluidState.getStateFromTQ(T_7, 1, self.params.orc_fluid).P_Pa
                    # p_7 = dp_loss_ass_cond + state[0].P_Pa

                # Desuperheater
                if self.params.dp_dT_loss[7] > 0:
                    p_6 = p_7 / (1 - self.params.dp_dT_loss[7])  # dp for the desuperheater
                else:
                    p_6 = self.params.dp_dT_loss[7] + p_7

                # Superheater
                if self.params.dp_dT_loss[4] > 0:
                    p_5 = state[3].P_Pa * (1 - self.params.dp_dT_loss[4])  # dp for the superheater
                else:
                    p_5 = state[3].P_Pa - self.params.dp_dT_loss[4]

                # Evaporator
                if self.params.dp_dT_loss[3] > 0:
                    p_3 = state[3].P_Pa / (1 - self.params.dp_dT_loss[3])  # dp for the boiler
                else:
                    T_3 = state[3].T_C + self.params.dp_dT_loss[4]  # - dT_sc
                    p_3 = FluidState.getStateFromTQ(T_3, 0, self.params.orc_fluid).P_Pa

                # Economizer
                if self.params.dp_dT_loss[2] > 0:
                    p_2 = p_3 / (1 - self.params.dp_dT_loss[2])  # dp for the pre-heater
                else:
                    p_2 = self.params.dp_dT_loss[2] + p_3

                # Compute TDN points
                # State 7 (Desuperheater -> Condenser)
                # saturated vapor
                state[6] = FluidState.getStateFromPQ(p_7, 1, self.params.orc_fluid)

                # State 5 (Superheater -> Turbine)
                T_5 = T_in_C - self.params.dT_ap_phe
                state[4] = FluidState.getStateFromPT(p_5, T_5, self.params.orc_fluid)

                # State 6 (Turbine -> Desuperheater)
                h_6s = FluidState.getStateFromPS(p_6, state[4].s_JK, self.params.orc_fluid).h_Jkg
                h_6 = state[4].h_Jkg - self.params.eta_turbine_orc * (state[4].h_Jkg - h_6s)
                state[5] = FluidState.getStateFromPh(p_6, h_6, self.params.orc_fluid)

                # State 3 (Preheater -> Boiler)
                # saturated liquid
                state[2] = FluidState.getStateFromPQ(p_3, 0, self.params.orc_fluid)

                # State 2 (Pump -> Preheater)
                h_2s = FluidState.getStateFromPS(p_2, state[0].s_JK, self.params.orc_fluid).h_Jkg
                h_2 = state[0].h_Jkg - ((state[0].h_Jkg - h_2s) / self.params.eta_pump_orc)
                state[1] = FluidState.getStateFromPh(p_2, h_2, self.params.orc_fluid)

                results = PowerPlantOutput()

                # Calculate orc heat/work
                w_pump_orc = state[0].h_Jkg - state[1].h_Jkg
                q_preheater_orc = -1 * (state[1].h_Jkg - state[2].h_Jkg)
                q_boiler_orc = -1 * (state[2].h_Jkg - state[3].h_Jkg)
                q_superheater_orc = -1 * (state[3].h_Jkg - state[4].h_Jkg)
                w_turbine_orc = state[4].h_Jkg - state[5].h_Jkg
                q_desuperheater_orc = -1 * (state[5].h_Jkg - state[6].h_Jkg)
                q_condenser_orc = -1 * (state[6].h_Jkg - state[0].h_Jkg)

                results.dP_pump_orc = state[1].P_Pa - state[0].P_Pa
                results.P_boil = state[2].P_Pa

            else:  #ciclo orc SH con rec

                # create empty list to compute cycle of 6 states
                state = [None] * 9

                # State 1 (Condenser -> Pump)
                # saturated liquid
                state[0] = FluidState.getStateFromTQ(T_condense_C, 0, self.params.orc_fluid)
                # p_1 = state[0].P_Pa

                # State 5 (Boiler -> Superheater)
                # saturated vapor
                state[4] = FluidState.getStateFromTQ(self.params.T_boil_C, 1, self.params.orc_fluid)

                # Compute dp and dT for each TDN points; relative if > 0, absolute if < 0
                # Condenser
                if self.params.dp_dT_loss[8] > 0:
                    p_9 = state[0].P_Pa / (1 - self.params.dp_dT_loss[8])  # dp for the condenser
                else:
                    T_9 = state[0].T_C + self.params.dp_dT_loss[8]
                    p_9 = FluidState.getStateFromTQ(T_9, 1, self.params.orc_fluid).P_Pa
                    # p_9 = dp_loss_ass_cond + state[0].P_Pa

                # Desuperheater
                if self.params.dp_dT_loss[7] > 0:
                    p_8 = p_9 / (1 - self.params.dp_dT_loss[7])  # dp for the desuperheater
                else:
                    p_8 = self.params.dp_dT_loss[7] + p_9

                # Evaporator
                if self.params.dp_dT_loss[3] > 0:
                    p_4 = state[4].P_Pa / (1 - self.params.dp_dT_loss[3])  # dp for the boiler
                else:
                    T_4 = state[4].T_C + self.params.dp_dT_loss[3]  # - dT_sc
                    p_4 = FluidState.getStateFromTQ(T_4, 0, self.params.orc_fluid).P_Pa

                # Preheater/Economizer
                if self.params.dp_dT_loss[2] > 0:
                    p_3 = p_4 / (1 - self.params.dp_dT_loss[2])  # dp for the pre-heater
                else:
                    p_3 = self.params.dp_dT_loss[2] + p_4

                # Superheater
                if self.params.dp_dT_loss[4] > 0:
                    p_6 = state[4].P_Pa * (1 - self.params.dp_dT_loss[4])  # dp for the superheater
                else:
                    p_6 = state[4].P_Pa - self.params.dp_dT_loss[4]

                # Recuperator hot side
                if self.params.dp_dT_loss[6] > 0:
                    p_7 = p_8 / (1 - self.params.dp_dT_loss[6])  # dp for the hot side recuperator
                else:
                    p_7 = self.params.dp_dT_loss[6] + p_8

                # Recuperator cold side
                if self.params.dp_dT_loss[1] > 0:
                    p_2 = p_3 * (1 - self.params.dp_dT_loss[1])  # dp for the cold side recuperator
                else:
                    p_2 = p_3 - self.params.dp_dT_loss[1]

                # Compute TDN points
                # State 9 (Desuperheater -> Condenser)
                # saturated vapor
                state[8] = FluidState.getStateFromPQ(p_9, 1, self.params.orc_fluid)

                # State 6 (Superheater -> Turbine)
                T_6 = T_in_C - self.params.dT_ap_phe
                state[5] = FluidState.getStateFromPT(p_6, T_6, self.params.orc_fluid)

                # State 7 (Turbine -> Recuperator)
                h_7s = FluidState.getStateFromPS(p_7, state[5].s_JK, self.params.orc_fluid).h_Jkg
                h_7 = state[5].h_Jkg - self.params.eta_turbine_orc * (state[5].h_Jkg - h_7s)
                state[6] = FluidState.getStateFromPh(p_7, h_7, self.params.orc_fluid)

                # State 4 (Preheater -> Boiler)
                # saturated liquid
                state[3] = FluidState.getStateFromPQ(p_4, 0, self.params.orc_fluid)

                # State 2 (Pump -> Recuperator)
                h_2s = FluidState.getStateFromPS(p_2, state[0].s_JK, self.params.orc_fluid).h_Jkg
                h_2 = state[0].h_Jkg - ((state[0].h_Jkg - h_2s) / self.params.eta_pump_orc)
                state[1] = FluidState.getStateFromPh(p_2, h_2, self.params.orc_fluid)

                # State 3 (Recuperator cold -> Preheater)
                T_3 = state[1].T_C + self.params.dT_pp_rec
                state[2] = FluidState.getStateFromPT(p_3, T_3, self.params.orc_fluid)

                # State 8 (Recuperator hot -> Desuperheater)
                h_8 = h_7 - state[3].h_Jkg + h_2
                state[7] = FluidState.getStateFromPh(p_8, h_8, self.params.orc_fluid)

                results = PowerPlantOutput()

                # Calculate orc heat/work
                w_pump_orc = state[0].h_Jkg - state[1].h_Jkg
                q_recuperator_orc = -1 * (state[1].h_Jkg - state[2].h_Jkg)
                q_preheater_orc = -1 * (state[2].h_Jkg - state[3].h_Jkg)
                q_boiler_orc = -1 * (state[3].h_Jkg - state[4].h_Jkg)
                q_superheater_orc = -1 * (state[4].h_Jkg - state[5].h_Jkg)
                w_turbine_orc = state[5].h_Jkg - state[6].h_Jkg
                q_desuperheater_orc = -1 * (state[7].h_Jkg - state[8].h_Jkg)
                q_condenser_orc = -1 * (state[8].h_Jkg - state[0].h_Jkg)

                results.dP_pump_orc = state[1].P_Pa - state[0].P_Pa
                results.P_boil = state[3].P_Pa

        # Cooling Tower Parasitic load
        dT_range = state[4].T_C - state[5].T_C
        parasiticPowerFraction = CoolingCondensingTower.parasiticPowerFraction(self.params.T_ambient_C,
                                                                               self.params.dT_approach, dT_range,
                                                                               self.params.cooling_mode)
        w_cooler_orc = q_desuperheater_orc * parasiticPowerFraction('cooling')
        w_condenser_orc = q_condenser_orc * parasiticPowerFraction('condensing')

        # water (assume pressure 100 kPa above saturation)
        P_sat = FluidState.getStateFromTQ(T_in_C, 0, 'Water').P_Pa
        cp = FluidState.getStateFromPT(P_sat + 100e3, T_in_C, 'Water').cp_JK
        # Water state 11, inlet, 12, mid, 13 exit
        T_C_11 = T_in_C
        T_C_12 = T_boil_C + dT_pinch
        # mdot_ratio = mdot_orc / mdot_water
        mdot_ratio = cp * (T_C_11 - T_C_12) / q_boiler_orc
        T_C_13 = T_C_12 - mdot_ratio * q_preheater_orc / cp

        # check that T_C(13) isn't below pinch constraint
        if T_C_13 < (state[1].T_C + dT_pinch):
            # pinch constraint is here, not at 12
            # outlet is pump temp plus pinch
            T_C_13 = state[1].T_C + dT_pinch
            R = q_boiler_orc / (q_boiler_orc + q_preheater_orc)
            T_C_12 = T_C_11 - (T_C_11 - T_C_13) * R
            mdot_ratio = cp * (T_C_11 - T_C_12) / q_boiler_orc

        # Calculate water heat/work
        results.q_preheater = mdot_ratio * q_preheater_orc
        results.q_boiler = mdot_ratio * q_boiler_orc
        results.q_desuperheater = mdot_ratio * q_desuperheater_orc
        results.q_condenser = mdot_ratio * q_condenser_orc
        results.q_superheater = mdot_ratio * q_superheater_orc
        results.q_recuperator = mdot_ratio * q_recuperator_orc
        results.w_turbine = mdot_ratio * w_turbine_orc
        results.w_pump = mdot_ratio * w_pump_orc
        results.w_cooler = mdot_ratio * w_cooler_orc
        results.w_condenser = mdot_ratio * w_condenser_orc
        results.w_net = results.w_turbine + results.w_pump + results.w_cooler + results.w_condenser

        # Calculate temperatures
        results.dT_range_CT = state[4].T_C - state[5].T_C
        dT_A_p = T_C_13 - state[1].T_C
        dT_B_p = T_C_12 - state[2].T_C
        if dT_A_p == dT_B_p:
            results.dT_LMTD_preheater = dT_A_p
        else:
            div = dT_A_p / dT_B_p
            results.dT_LMTD_preheater = (dT_A_p - dT_B_p) / (math.log(abs(div)) * np.sign(div))

        dT_A_b = T_C_12 - state[2].T_C
        dT_B_b = T_C_11 - state[3].T_C
        if dT_A_b == dT_B_b:
            results.dT_LMTD_boiler = dT_A_b
        else:
            div = dT_A_b / dT_B_b
            results.dT_LMTD_boiler = (dT_A_b - dT_B_b) / (math.log(abs(div)) * np.sign(div))

        # return temperature
        results.state = FluidState.getStateFromPT(initialState.P_Pa, T_C_13, self.params.working_fluid)

        return results
