##  Purpose: Modeling surface residue decomposition and N release
##  Authors:Eunjin Han, USDA-ARS-ACSL, March, 2025
##  Note: This model is based on 1)Thapa, Resham, et al. "Modeling surface residue decomposition and N release using the cover crop nitrogen calculator (CC-NCALC)."
#   Nutrient Cycling in Agroecosystems 124.1 (2022): 81-99.
##    2) water potential model is from Dann, Carson E., et al. "Modeling water potential of cover crop residues on the soil surface." Ecological Modelling 459 (2021): 109708.e

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from datetime import datetime, timedelta
import os
from scipy.optimize import fsolve
# ----------------------- Global Variables -------------------------
# These variables are equivalent to the variables in the Fortran INCLUDE files
# (e.g., public.ins, puplant.ins, etc.).
# You'll need to initialize these with appropriate values or data structures.
# ==================variable used in mulch decomposition module => took from Z's mulch decomposition module in 2DSOIL
# c     FLOAT TYPE  [UNIT]
# c         PC_mulch        [g/g]   Mass concentration of C in mulch, g of C/g of solid portion of mulch, should be 0-1, from literature, e.g. in (https://www.pnas.org/content/115/16/4033), PC=0.4368
# c         PN_mulch        [g/g]   Mass concentration of N in mulch, g of N/g of solid portion of mulch, should be 0-1, from literature, e.g. in (https://www.pnas.org/content/115/16/4033), PN=0.0140
# c         alpha_F         [1]     The mulch feeding parameter, if the decomposition on soil-mulch interface is "a", then the decompositon for one layer above should be "alpha_mulch*a"
# c         Frac_CARB       [1]     Fraction of Carbohydrate over Total Organic Mulch, should be 0-1, change during the simulation
# c         Frac_CELL       [1]     Fraction of Holo-Cellulose over Total Organic Mulch, should be 0-1, change during the simulation
# c         Frac_LIGN       [1]     Fraction of Lignin over Total Organic Mulch, should be 0-1, change during the simulation
# c         Frac_CARB_Init  [1]     Fraction of Initial Carbohydrate over Total Organic Mulch, should be 0-1, fixed before chemical decomposition
# c         Frac_CELL_Init  [1]     Fraction of Initial Holo-Cellulose over Total Organic Mulch, should be 0-1, fixed before chemical decomposition
# c         Frac_LIGN_Init  [1]     Fraction of Initial Lignin over Total Organic Mulch, should be 0-1, fixed before chemical decomposition 
# c         FracN_CARB_Init [1]     Fraction of Initial N in Carbohydrate over Carbohydrate, should be 0-1, fixed before chemical decomposition
# c         FracN_CELL_Init [1]     Fraction of Initial N in Holo-Cellulose over Holo-Cellulose, should be 0-1, fixed before chemical decomposition
# c         FracN_LIGN_Init [1]     Fraction of Initial N in Lignin over Lignin, should be 0-1, fixed before chemical decomposition 
# c         Humid_Factor    [1]     Humification factor, 0.125D0
# c         K_CARB          [day-1] Coef of decompositon rate, reference factor for CARB, need adjusted
# c         K_CELL          [day-1] Coef of decompositon rate, reference factor for CELL, need adjusted
# c         K_LIGN          [day-1] Coef of decompositon rate, reference factor for LIGN, need adjusted
# c         K_CARB_temp     [day-1] Coef of decompositon rate for CARB, after adjusted
# c         K_CELL_temp     [day-1] Coef of decompositon rate for CELL, after adjusted
# c         K_LIGN_temp     [day-1] Coef of decompositon rate for LIGN, after adjusted
# c         mulch_mass_temp [g]     All the availiable soild mulch in the target mulch element
# c         N_mass_temp     [g]     All the availiable N in the target mulch element
# c         INKG            [g]     Microbe-available inorganic N in shallow soil layer
# c         CNR             [1] the universal C/R ratio for an given element 
# c         Frac_Decomp(Layer)  [1] the decomposition fraction of each layer, 0 means no decomposition, 1 means totally gone
# c         CO2mass_2_Cmass     [1] convert CO2 mass to C mass, 12/44
# c         NO3mass_2_Nmass     [1] convert NO3 mass to N mass, 14/62  
# c         NH4mass_2_Nmass     [1] convert NH4 mass to N mass, 14/18
# c         ThetaV_2_ThetaM     [1] convert volumetric water content to gravimetric water content, there ThetaV is based on the whole mulch not based on mulch solid, so should be devided by (1-f_mulch_pore) first, then use mulch bulk density
# c         HNew_m          [MPa]   Mulch water potential
# c         Theta_m         [g/g]   Mulch gravimetric water content, refer to the total mulch
# c         Temp_m          [oC]    Mulch temperature
# c
# c ------------------------mass pools---------------------------------------------
# c         CARB_mass       [g]     CARB mass of the mulch, CARB_mass+CELL_mass+LIGN_mass=mulch_mass_temp
# c         CELL_mass       [g]     CELL mass of the mulch, CARB_mass+CELL_mass+LIGN_mass=mulch_mass_temp
# c         LIGN_mass       [g]     LIGN mass of the mulch, CARB_mass+CELL_mass+LIGN_mass=mulch_mass_temp
# c         CARB_N_mass     [g]     N within CARB, CARB_N_mass+CELL_N_mass+LIGN_N_mass=N_mass_temp
# c         CELL_N_mass     [g]     N within CELL, CARB_N_mass+CELL_N_mass+LIGN_N_mass=N_mass_temp
# c         LIGN_N_mass     [g]     N within LIGN, CARB_N_mass+CELL_N_mass+LIGN_N_mass=N_mass_temp
# c
# c ------------------------decomposition rate---------------------------------------------
# c         CNRF            [0-1]   CNR effects on decomposition
# c         MTRF            [0-1]   Moisture and Temperature effects on decomposition
# c         N_im            [g/day] the immobilization of N before N reach the soil surface (N interception by mulch)
# c         CARB_Decomp     [g/day] Decomposition speed for CARB, after add CNRF and MTRF
# c         CELL_Decomp     [g/day] Decomposition speed for CELL, after add CNRF and MTRF
# c         LIGN_Decomp     [g/day] Decomposition speed for LIGN, after add CNRF and MTRF
# c         FOM_Decomp      [g/day] Decomposition speed for all the organic matter (mulch residue)
# c         CARB_Decomp_N   [g/day] Gross Decomposition speed for N in CARB, after add CNRF and MTRF
# c         CELL_Decomp_N   [g/day] Gross Decomposition speed for N in CELL, after add CNRF and MTRF
# c         LIGN_Decomp_N   [g/day] Gross Decomposition speed for N in LIGN, after add CNRF and MTRF
# c         FOMN_Decomp     [g/day] Gross/Actual Decomposition speed for N in organic matter (mulch residue N)
# c         CARB_total_decomp_R   [g] Total residue decomposition of CARB for mulch-soil interface
# c         CELL_total_decomp_R   [g] Total residue decomposition of CELL for mulch-soil interface
# c         LIGN_total_decomp_R   [g] Total residue decomposition of LIGN for mulch-soil interface
# c         CARB_total_decomp_N   [g] Total gross N mineralization of CARB for mulch-soil interface  
# c         CELL_total_decomp_N   [g] Total gross N mineralization of CELL for mulch-soil interface 
# c         LIGN_total_decomp_N   [g] Total gross N mineralization of LIGN for mulch-soil interface
# c         Nim_total_decomp      [g] Total N mulch-interception
# c         Nmine_total_decomp    [g] Total N mineralization, >0 mulch to soil; <0 soil to mulch
# c         Nhumi_total_decomp    [g] Total N humification, >0 mulch to soil
# c         Mulch_Decompose [g]     Total decomposition on mulch-soil interface based on MULCH (NOT C-BASED HERE)      
# c         Mulch_Avail     [g]     Total availiable mulch for the current layer (horizontal level)
# c         Mulch_AvailUp   [g]     Total availiable mulch for the current layer (horizontal level) when considering a two layered structure
# c         Mulch_AvailDown [g]     Total availiable mulch for the lower layer (horizontal level) when considering a two layered structure
# c         totalMulchWidth [cm]    the whole mulch width
# c         CO2_to_Air      [nu g CO2/cm^3] the concentration of CO2 added to air based on mulch decomposition and mulch volume.


# ----------------------- Initialization Functions -------------------------

# def read_mulch_input(mulch_file):
#     """
#     Reads mulch decomposition parameters from a file.

#     Args:
#         mulch_file (str): The path to the mulch input file.

#     Returns:
#         None: Initializes global variables.
#     """
#     try:
#         with open(mulch_file, 'r') as f:
#             for _ in range(3):
#                 in_string_mulch = f.readline().strip()
#                 if _ == 0 and in_string_mulch[0:23] != '[Mulch_Decomposition]':
#                     continue

#             global Crit_Level, alpha_F
#             Crit_Level, alpha_F = map(float, f.readline().split())
#             for _ in range(3):
#                 f.readline()

#             global Frac_CARB_Init, Frac_CELL_Init, Frac_LIGN_Init
#             Frac_CARB_Init, Frac_CELL_Init, Frac_LIGN_Init = map(float, f.readline().split())
#             for _ in range(3):
#                 f.readline()

#             global FracN_CARB_Init, FracN_CELL_Init, FracN_LIGN_Init
#             FracN_CARB_Init, FracN_CELL_Init, FracN_LIGN_Init = map(float, f.readline().split())
#             for _ in range(3):
#                 f.readline()

#             global K_CARB, K_CELL, K_LIGN
#             K_CARB, K_CELL, K_LIGN = map(float, f.readline().split())

#     except FileNotFoundError:
#         print('Mulch Information is not Included')
#         return

# def initialize_mulch_data():
#     """
#     Initializes mulch data structures and sets up critical levels.
#     This corresponds to the Fortran code under `IF(MulchDecompIni.eq.1 ...`.

#     Args:
#         None: Uses global variables.

#     Returns:
#         None: Initializes mulch-related data structures (e.g., arrays).
#     """

#     global LocalFlag_MulchDecomp_Start, LocalFlag_MulchDecomp_FinalAssign, Crit_Level
#     LocalFlag_MulchDecomp_Start = 1
#     LocalFlag_MulchDecomp_FinalAssign = 0

#     Crit_Level = Crit_Level * mulchThick  #  mulchThick needs to be defined

#     global Frac_CARB, Frac_CELL, Frac_LIGN, ThetaV_2_ThetaM
#     Frac_CARB = Frac_CARB_Init
#     Frac_CELL = Frac_CELL_Init
#     Frac_LIGN = Frac_LIGN_Init

#     ThetaV_2_ThetaM = 1.0 / (rho_mulch_b / rho_w)  # rho_mulch_b, rho_w  need to be defined

#     global CARB_mass, CELL_mass, LIGN_mass, CARB_N_mass, CELL_N_mass, LIGN_N_mass
#     # Initialize mass pools (NumPy arrays)
#     CARB_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))  # mulchLayer, SurNodeIndex_M need to be defined
#     CELL_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))
#     LIGN_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))
#     CARB_N_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))
#     CELL_N_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))
#     LIGN_N_mass = np.zeros((mulchLayer, SurNodeIndex_M - 1))

#     for kk in range(SurNodeIndex_M - 1):
#         for jj in range(mulchLayer):
#             EleIndex = MulchEleMatrix[jj, kk]  # MulchEleMatrix needs to be defined
#             mulch_mass_temp = MulchEleMarkerArea[EleIndex] * rho_mulch_b * 1.0e-6  # MulchEleMarkerArea needs to be defined
#             aaaa = mulch_mass_temp * Frac_CARB_Init
#             CARB_mass[jj, kk] = aaaa
#             CARB_N_mass[jj, kk] = aaaa * FracN_CARB_Init
#             aaaa = mulch_mass_temp * Frac_CELL_Init
#             CELL_mass[jj, kk] = aaaa
#             CELL_N_mass[jj, kk] = aaaa * FracN_CELL_Init
#             aaaa = mulch_mass_temp * Frac_LIGN_Init
#             LIGN_mass[jj, kk] = aaaa
#             LIGN_N_mass[jj, kk] = aaaa * FracN_LIGN_Init

#     global Nim_cumu_decomp, Nmine_cumu_decomp, Nhumi_cumu_decomp
#     Nim_cumu_decomp = 0.0
#     Nmine_cumu_decomp = 0.0
#     Nhumi_cumu_decomp = 0.0

# # ----------------------- Calculation Functions -------------------------
#EJ: to estimate rain amount equivalent to the inital residue gravemetric water content (theta_g) at the onset of rain event
def estimate_x_given_y(func, y_value, x_guess):  
    """
    Estimates the value of x for a given y in the function func(x).
    => fsolve from scipy.optimize is employed to find the root of the equation func(x) - y_value = 0
        , which corresponds to the estimated x for the given y.
    Args:
        func (callable): The function to solve for x.
        y_value (float): The target y value.
        x_guess (float): An initial guess for x.

    Returns:
        float: The estimated x value.
    """

    def equation_to_solve(x):
        return func(x) - y_value

    estimated_x = fsolve(equation_to_solve, x_guess)[0]
    return estimated_x


def calculate_decomposition():
    """
    Calculates mulch decomposition, N immobilization, and related processes.
    This function contains the main calculation logic of the Fortran code.

    Args:
        None: Uses and modifies global variables.

    Returns:
        None: Updates global variables with decomposition results.
    """  
    global mulch_mass_init, N_inorg_init
    global Frac_CARB_Init, Frac_CELL_Init, Frac_LIGN_Init, FracN_CARB_Init, FracN_CELL_Init, FracN_LIGN_Init
    global K_CARB_coeff, K_CELL_coeff, K_LIGN_coeff, base_T
    global a_rain, b_rain, c_rain, d_rain, param_a, param_b, param_k2_c
    global a_MTRF, b_MTRF, c_MTRF, d_MTRF, a_CNRF, CNR_crit, RM_min, RM_ma
    global Microbial_N_dmd, HUMF
    global Init_date, End_date, weather_fname, out_fname
    global residue_Theta_g

    #*parameters for computing air weater potential
    # Ψ = -RTln(HR), where Ψ is water potential (MPa), 
    # R is the universal gas constant (8.3143 J mol⁻¹ K⁻¹), 
    # T is the absolute temperature (Kelvin), and RH is the relative humidity (unitless). 
    R_gas = 8.3143 # J mol⁻¹ K⁻¹
    V_air = 1.8 * 10**(-5) #partial molar volume of water
    
    #initial values => adopted from Z's fortran code
    aaaa = mulch_mass_init * Frac_CARB_Init  # kg/ha 
    CARB_mass = aaaa
    CARB_N_mass = aaaa * FracN_CARB_Init   # kg N/ha 
    aaaa = mulch_mass_init * Frac_CELL_Init
    CELL_mass = aaaa
    CELL_N_mass = aaaa * FracN_CELL_Init  # kg N/ha 
    aaaa = mulch_mass_init * Frac_LIGN_Init
    LIGN_mass = aaaa
    LIGN_N_mass = aaaa * FracN_LIGN_Init

    cum_RM_N_Decomp = 0.0 #initialize
    cum_net_RM_N_Decomp = 0.0 #initialize
    new_theta_g = 0.0

    ########################################################
    # Determine residue water potential based on Dann (2021)
    ########################################################
    #initial values
    old_RH = 0.8  #RH [0-1] at 11pm on previous day (before the Init_date) => arbitrary positive number 
    
    # 1)read hourly weather data
    # weather_fname = r'D:\ACSL_EJ\FSP_paper\Residue_decom_modeling\Python_model\FSPWeather_1996_2021_CO2.csv'
    df = pd.read_csv(weather_fname, parse_dates=["date"])  
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'Hour', 'temperature', 'rain', 'RH']]
    df['datetime'] = df.apply(lambda row: row['date'] + pd.Timedelta(hours=row['Hour']), axis=1) # Add days from 'days_to_add' to 'date'
    df.loc[df["RH"] > 100, "RH"] = 100 # Replacing RH> 100 with RH=100
    df['RH']= df['RH']*0.01  #RH expressed as a fraction [0-1], not % according to Dann (2021)
    df['month']= df['date'].dt.month
    df['day']= df['date'].dt.day
    # Filter weather for the target period
    df = df[(df['date'] >= Init_date) & (df['date'] <= End_date)]
    df = df.reset_index()
    print(df)

    current_date = Init_date 
    current_hour = current_date.hour

    #create an empty pandas dataframe 
    column_names = ['datetime', 'rain','temp','RH', 'CARB_mass', 'CELL_mass', 'LIGN_mass', 'CARB_N_mass', 'CELL_N_mass', 'LIGN_N_mass', 
                    'N_mineralized', 'CUMN_mineralized','N_immobilized','soil_inorgN','RM_T', 'RM_N', 'kCARB', 'kCELL', 'kLIGN', 'MTRF', 
                    'CNRF', 'ContactFactor', 'micro_N_demand', 'GrossN_minerlized']
    df_out = pd.DataFrame(columns=column_names)


    while current_date <= End_date:  #houlry loop
        print(f"current date= ' {current_date}")
        #1)compute updated residue water content depending on rain
        current_rain = df[(df['datetime'] == current_date) & (df['Hour'] == current_hour)].rain.values[0] #check
        current_T = df[(df['datetime'] == current_date) & (df['Hour'] == current_hour)].temperature.values[0] #check
        current_RH = df[(df['datetime'] == current_date) & (df['Hour'] == current_hour)].RH.values[0] #chould be computed regardless of rain event to update old_RH later
        if current_rain > 0:  #rain effect on Theta_g of cereal ry eresidue => eqn (7) in Dann (2021)
            def Theta_rain_fn(rain_x): #rain effect on theta_g as a fn of rainfall amount [mm/hr] + rain equivalent to the initial theta_g
                return a_rain*(1-np.exp(b_rain*rain_x)) + c_rain*(1-np.exp(d_rain*rain_x)) 
                # return 0.8523*(1-np.exp(-0.9523*rain)) + 2.9558*(1-np.exp(-0.0583*rain)) 
            target_y = residue_Theta_g
            initial_guess_x = 10  #range [1- 50 mm/hr]
            rain_equv_init_theta = estimate_x_given_y(Theta_rain_fn, target_y, initial_guess_x)  #rain equivalent to the initial theta_g

            print(f"Estimated rain amount for theta_g = {target_y}: {rain_equv_init_theta}")
            #new_theta_g => updated residue_theta_g
            new_theta_g = a_rain*(1-np.exp(b_rain*(current_rain + rain_equv_init_theta))) + c_rain*(1-np.exp(d_rain*(current_rain + rain_equv_init_theta))) 
            print(f"increase in thetat_g = {new_theta_g - residue_Theta_g}")
        else: #no rain => compute residue water content using RH and gradient of water potential betwen air and residue
            psi_air = R_gas * (current_T + 273.5) * np.log(current_RH) /(V_air * 10**(6))  #Eqn(2) of Dann (2021). 10^6 is to convert Pa to MPa
            psi_residue = param_a* (residue_Theta_g**param_b)  #check
            psi_gradient = psi_air - psi_residue #check
            psi_gradient = max([psi_air - psi_residue, -200]) #a limit of -200 Mpa was placed to prevent excessive decreases in residue theta_g (see Dann 2021)
            delta_RH = current_RH - old_RH
            if delta_RH >= 0:  #wetting - transfer water from the air to the residue
                if psi_gradient < 30 and delta_RH > 0:
                    k1=0.0008 #eqn (8) in Dann(2021)  => may need to adjust for maize/sb stover
                elif psi_gradient >= 30 and delta_RH > 0:
                    k1=0.0004 #eqn (8) in Dann(2021)
                else: #del_RH <= 0
                    k1=0
                print(f"del_residue theta_g = k1*psi_grad = {k1} * {psi_gradient} = {k1*psi_gradient}")
                # del_theta_g = max([k1*psi_gradient, 0.02]) #eqn (8) in Dann(2021) if k1*psi_grad > 0.01 => may need to adjust for maize/sb stover
                if k1*psi_gradient > 0.01:   #0.02?:  #why cap at0.01? Maybe typo??? 0.02 was possible with dry residue
                    print("k1*psi_gradient > 0.01   !check")
                    del_theta_g = 0.02    # why 0.02?? I think 0.02 was used as a cap, but if 0.02, there is a jump
                # del_theta_g = min([k1*psi_gradient, 0.01])  #I think 0.01 is a typo....
                del_theta_g = min([k1*psi_gradient, 0.02])
            else:   #drying  => eqn (8) and (9) in Dann (2021)
                k2 = param_k2_c * current_T * residue_Theta_g
                print(f"k2 = {k2} during drying/evaporation")
                if residue_Theta_g < 0.04 and delta_RH >= 0:
                    del_theta_g = 0
                elif residue_Theta_g > 0.04 and delta_RH > -2.5 and delta_RH < 0: #EJ added "and delta_RH < 0"
                    del_theta_g = 0.001
                elif residue_Theta_g > 0.04 and delta_RH <= -2.5 and k2 >= 0.03:
                    print(f"k2 = {k2} during drying/evaporation if k2> 0.03")
                    del_theta_g = 0.085
                elif residue_Theta_g > 0.04 and delta_RH <= -2.5 and k2 < 0.03:
                    del_theta_g = k2
                else:
                    print('else?? in computing residue_Theta_g during drying')

            #update current residue theta_g
            new_theta_g = residue_Theta_g + del_theta_g  #del_theta_g could be positive (from wetting) or negative (from drying)
            
            # Transfer of water into the residue as a result of dew occurs once a day
            # when there is no rain and the RH(t) exceeds 85%
            # if the residue theta_g is below 2.5 and residue biomass exceeds 1500 kg/ha, 
            # due deposition increases theta_g of cereal rey by 0.49
            if current_hour == 6 and current_rain == 0 and current_RH > 0.85:  #check current hour
                new_theta_g = residue_Theta_g + 0.49

        #============ update residue water potential    
        new_psi_residue = param_a* (new_theta_g**param_b)  #check

        #============ Now, compute residue decomposition
        RM_T = LIGN_mass + CELL_mass + CARB_mass
        Frac_LIGN = LIGN_mass/RM_T #lignin % in the remaining total residue
        k_LIGN = K_LIGN_coeff
        k_CARB = K_CARB_coeff * np.exp(-12.0 * Frac_LIGN)  #eqn (2) in Thapa(2022)
        k_CELL = K_CELL_coeff * np.exp(-12.0 * Frac_LIGN)  
        
        if current_T > 0:  #temperature
            MTRF = (a_MTRF + b_MTRF * current_T) * np.exp((c_MTRF + d_MTRF*current_T**(-1))*new_psi_residue)
        else:
            MTRF = 0.0
    
        CNR = PC_mulch * RM_T/(CARB_N_mass + CELL_N_mass + LIGN_N_mass + N_inorg_init) #check should be 
        if CNR > CNR_crit:  #CNR_crit =13
            CNRF = np.exp(a_CNRF*(CNR - CNR_crit)/CNR_crit)
        else:
            CNRF = 1.0
        
        #contact factor
        if RM_T <= RM_min:  #RM_min= 1400
            ContactFactor = 1
        elif RM_T > RM_min and RM_T <= RM_max:  #RM_max = 3000 
            ContactFactor = RM_min/RM_T
        else:  #RM_T > RM_max:  #RM_max = 3000 
            ContactFactor = RM_min/RM_max
        
        #The hourly amount of residue mass decomposed [kg/ha/hr]  <= eqn (1) in Thapa(2022)
        CARB_Decomp = k_CARB * CARB_mass * MTRF * CNRF * ContactFactor #! [kg/ha/hr] 
        CELL_Decomp = k_CELL * CELL_mass * MTRF * CNRF * ContactFactor 
        LIGN_Decomp = k_LIGN * LIGN_mass * MTRF * CNRF * ContactFactor
        RM_T_Decomp = CARB_Decomp + CELL_Decomp + LIGN_Decomp  #Total Residue mass decomposed

        #Gross N mineralized from surface residue [kg N/ha/hr] <= eqn (1) in Thapa(2022)
        CARB_Decomp_N = k_CARB * CARB_N_mass * MTRF * CNRF * ContactFactor #! [kg N/ha/hr] 
        CELL_Decomp_N = k_CELL * CELL_N_mass * MTRF * CNRF * ContactFactor 
        LIGN_Decomp_N = k_LIGN * LIGN_N_mass * MTRF * CNRF * ContactFactor
        RM_N_Decomp = CARB_Decomp_N + CELL_Decomp_N + LIGN_Decomp_N  #! the gross value from the above three

        #convert the gross decomposition to net decomposition        
        #==========Nitrogen immobilization similar to CERES-N but set to occur when microbial N demand during residue decomposition
        #                   is not fulfilled from gross N mineralized
        N_im = min([RM_T_Decomp * Microbial_N_dmd - RM_N_Decomp, N_inorg_init])      #Microbial_N_dmd = 0.0213 
        # N_im = max([N_im, 0.0])      ###???      RM_T_Decomp * Microbial_N_dmd - RM_N_Decomp can be negative....
        #update soil N if N immobilization occurs
        temp = RM_T_Decomp * Microbial_N_dmd - RM_N_Decomp
        if temp > 0: #icrobial N demand during residue decomposition is NOT met => take N from soil (i.e., immobilization) and add to CARB_N pool
            N_inorg_init = N_inorg_init - temp
            CARB_N_mass = CARB_N_mass + temp
            net_RM_N_Decomp = 0.0
            #============ update variables for a next time step (hour)  
            CARB_mass = CARB_mass - CARB_Decomp
            CELL_mass = CELL_mass - CELL_Decomp
            LIGN_mass = LIGN_mass - LIGN_Decomp

            # CARB_N_mass = CARB_N_mass - CARB_Decomp_N
            CELL_N_mass = CELL_N_mass - CELL_Decomp_N
            LIGN_N_mass = LIGN_N_mass - LIGN_Decomp_N
        else:  #enough N is mineralized from residue to meet microbial N demand and humification
            net_RM_N_Decomp = RM_N_Decomp* (1-HUMF) - N_im  #Net N mineralized from surface residue            
            N_Humi = RM_N_Decomp * HUMF #12.5% of gross N mineralized from surface residue is synthesized back into soil OM 

            #============ update variables for a next time step (hour)  
            CARB_mass = CARB_mass - CARB_Decomp
            CELL_mass = CELL_mass - CELL_Decomp
            LIGN_mass = LIGN_mass - LIGN_Decomp

            CARB_N_mass = CARB_N_mass - CARB_Decomp_N
            CELL_N_mass = CELL_N_mass - CELL_Decomp_N
            LIGN_N_mass = LIGN_N_mass - LIGN_Decomp_N

            
        RM_T = LIGN_mass + CELL_mass + CARB_mass  #total remaining residue mass
        RM_N = LIGN_N_mass + CELL_N_mass + CARB_N_mass  #total remaining N mass in residue
        # cum_RM_N_Decomp = cum_RM_N_Decomp + RM_N_Decomp  #cumulative N mineralized from surface residue
        cum_net_RM_N_Decomp = cum_net_RM_N_Decomp + net_RM_N_Decomp  #cumulative N mineralized from surface residue

        #update soil inorgnaic N pool
        # N_inorg_init = N_inorg_init + net_RM_N_Decomp  #=> this is accumulating too much soil N
        N_inorg_init = N_inorg_init + net_RM_N_Decomp*0.1  #=> this is accumulating too much soil N


        # # Ensure that the loop does not run past the end_time
        # if current_date >= End_date:
        #     break

        #=================update output df
        df_out.loc[len(df_out)] = [current_date, current_rain, current_T, current_RH,CARB_mass, CELL_mass, LIGN_mass,
                                    CARB_N_mass, CELL_N_mass, LIGN_N_mass, net_RM_N_Decomp, cum_net_RM_N_Decomp, N_im, N_inorg_init,
                                   RM_T, RM_N, k_CARB, k_CELL, k_LIGN, MTRF, CNRF,ContactFactor, RM_T_Decomp * Microbial_N_dmd, RM_N_Decomp] 
        # column_names = ['datetime', 'rain','temp','RH', 'CARB_mass', 'CELL_mass', 'LIGN_mass', 'CARB_N_mass', 'CELL_N_mass', 'LIGN_N_mass', 
        #             'N_mineralized', 'CUMN_mineralized','N_immobilized','soil_inorgN','RM_T', 'RM_N', 'kCARB', 'kCELL', 'kLIGN', 'MTRF', 'CNRF', 'ContactFactor']
    

        current_date = current_date + timedelta(hours=1)  #check
        current_hour = current_date.hour
        print(current_hour)
        residue_Theta_g = new_theta_g
        old_RH = current_RH
    
    #write output df into csv                                                                                                       
    df_out.to_csv(out_fname, index=False)
    #====================end
# ----------------------- Main Program Flow ------------------------

if __name__ == "__main__":
    # Initialize global variables and data structures
    # # ... (You'll need to define and initialize these) ...
    # lInput = 1 #replace with actual condition
    # MulchDecompIni = 1 #replace with actual condition
    # BoolMulch_TotalDecomposed = 0 #replace with actual condition
    # MulchFile = 'mulch_input.txt'  # Example mulch input file
    #====3/17/2025: input parameters for fresh rye residue => need to be changed appropriately for corn/soybean stover/residue decomposition
    mulch_mass_init = 5000 # kg/ha -> 150-10000 in Fig 4 Thampa (2022), 12000Table 2 in Wang (2021)
    residue_Theta_g = 1  # initial residue gravimetric water content [g H2O/g dry matter]  => See Fig. 4 and eqn (7) in Dann (2021) and 
    Frac_CARB_Init = 0.2  #see the resonable range in Table 1 in Thapa (2022)
    Frac_CELL_Init = 0.7  #see the resonable range in Table 1 in Thapa (2022)
    Frac_LIGN_Init = 0.1   #see the resonable range in Table 1 in Thapa (2022)
    FracN_CARB_Init = 0.01  #0.08
    FracN_CELL_Init = 0.01
    FracN_LIGN_Init = 0.01
    K_CARB_coeff = 0.018 #[h^(-1)] from Table 2 in Thapa(2022)   #Equn (2) in Thapa (2022) (0.43 [day^[-1] in Table 2 in Wang (2021)
    K_CELL_coeff = 0.010 #[h^(-1)] from Table 2 in Thapa(2022)   #Equn (2) in Thapa (2022) (0.24 [day^[-1] in Table 2 in Wang (2021)
    K_LIGN_coeff = 0.00095  #[h^(-1)] from Table 2 in Thapa(2022)   #Equn (2) in Thapa (2022) (0.0228 [day^[-1] in Table 2 in Wang (2021)
    base_T = 0  #[C] base temperature for decomposition from Table 2 in Thapa(2022) 

    #empirical coefficient for the relationship betwen residue water content (theta_g) and rainfall eqn(7) in Dann(2021)
    # theta_g = a(1-exp(b*rain))+ c*(1-exp(d*rain))
    a_rain = 0.8523  #for rye specific => might need to change for other crop residues
    b_rain = -0.9523  #for rye specific => might need to change for other crop residues
    c_rain = 2.9558   #for rye specific => might need to change for other crop residues
    d_rain = -0.0583   #for rye specific => might need to change for other crop residues

    #parameters of equation describing water release curve (psi_residue = a*(residue theta_g)^(b))
    # as a fn of residue lignin content  => see Fig. 2 in Dann (2021)
    param_a = -8.1651 + 0.5951* Frac_LIGN_Init*100  #for rye specific => might need to change for other crop residues
    param_b = -0.3752 - 0.1182* Frac_LIGN_Init*100
    # psi_residue = param_a* (residue_Theta_g**param_b)  #check
    
    #parameter to determine delta theta_g during drying
    param_k2_c = 0.01   #eqn (6) and (11) in Dann (2021)

    #parameter to adjust residue decomposition by environment (residue water potential and temperature)
    a_MTRF = 0.384    #eqn (3) in Thapa(2022)
    b_MTRF = 0.018    
    c_MTRF = 0.142    
    d_MTRF = 0.628    

    a_CNRF = -0.693
    CNR_crit = 13 #Critical C:N ratio,  from Table 2 in Thapa(2022)
    RM_min = 1400  #critical residue mass in direct soil contact (RM_min [kg/ha]),  from Table 2 in Thapa(2022)
    RM_max = 3000  #optimal residue mass above which decompostion is no longer impacted(RM_max [kg/ha]),  from Table 2 in Thapa(2022)

    # CNR represents the overal CN ratio in the contacting portion rather than the CN ratio of individual residue pools
    # CNR = 0.41 *RM_total/(RM_N + N_inorg_init), where  
    N_inorg_init = 5 #kg N/ha **** check => arbitrary number (note: 15ppm N in 5 cm soil=> 9 kg N/ha )
                     # in Table 1 Thapa (2022), Total N content (RN0) in residue are in range of 6-165 or 10-240 kg N/ha
# c         PC_mulch        [g/g]   Mass concentration of C in mulch, g of C/g of solid portion of mulch, should be 0-1, from literature, e.g. in (https://www.pnas.org/content/115/16/4033), PC=0.4368
# c         PN_mulch        [g/g]   Mass concentration of N in mulch, g of N/g of solid portion of mulch, should be 0-1, from literature, e.g. in (https://www.pnas.org/content/115/16/4033), PN=0.0140
    PC_mulch=0.426  #Z used 4.1 ! Concentrations (https://www.pnas.org/content/115/16/4033)
                    # Assume stover contains 40 percent C. https://crops.extension.iastate.edu/encyclopedia/carbon-and-nitrogen-cycling-corn-biomass-harvest
    # PN_mulch=0.01D0                   ! Concentrations (https://www.pnas.org/content/115/16/4033)
    
    #Microbial N demand (g N required per g RM decomposed) => see Thapa (2022)
    # => computed by multiplying the fraction of C in residues (0.426 g/g) by the microbial C use efficiency (0.4 g assimilated per g C FOM decomposed)
    #    and dividing by the microbial biomass C:N (i.e., 8) =? 0.4*0.426/8
    Microbial_N_dmd = 0.0213 
    #Humification factor
    HUMF = 0.125 #12.5% of gross N mineralized from surface residue is synthesized back into soil OM or to the more stable pool (i.e., FractionSynthesized) 
                 #and the rest enters the soil inorganic N pool (Woodruff et al. 2018)

    Init_date = datetime(2005, 5, 1)  #'05/01/2005'  D:\ACSL_EJ\FSP_paper\Residue_decom_modeling\Python_model\2005_103_NTSB    
    End_date = datetime(2005, 10, 3)  # '10/03/2005'
    
    # weather_fname = r'D:\ACSL_EJ\FSP_paper\Residue_decom_modeling\Python_model\FSPWeather_1996_2021_CO2.csv'
    weather_fname = r'D:\ACSL_EJ\FSP_paper\Residue_decom_modeling\Python_model\FSPWeather_2005.csv'
    out_fname = r'D:\ACSL_EJ\FSP_paper\Residue_decom_modeling\Python_model\hourly_output.csv'
    
    calculate_decomposition()

    # output_results()

    # if BoolMulch_TotalDecomposed == 1 and LocalFlag_MulchDecomp_FinalAssign == 0:
    #     final_assignment()
    # # ... (You might have other function calls or logic here) ...