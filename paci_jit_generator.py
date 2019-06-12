import numpy as np
from jitcode import jitcode, y
from symengine import exp, log, sqrt, tanh, sympify

def wrapper():
    tDrugApplication = 10000
    INaFRedMed = 1
    ICaLRedMed = 1
    IKrRedMed = 1
    IKsRedMed = 1

    return Paci2018(tDrugApplication, INaFRedMed,
             ICaLRedMed, IKrRedMed, IKsRedMed)

sigmoid = lambda x: (tanh(x)+1)/2

SHARPN_COEFF = 1e5
def conditional(trigger,threshold,untriggered_value,triggered_value,sharpness=None):
    if sharpness is None:
        if sympify(threshold).is_number and threshold:
            sharpness = SHARPN_COEFF*abs(threshold)
        else:
            sharpness = SHARPN_COEFF
    
    return untriggered_value + sigmoid((trigger-threshold)*sharpness)*(triggered_value-untriggered_value)

def Paci2018(tDrugApplication, INaFRedMed,ICaLRedMed, IKrRedMed, IKsRedMed):
    dY = [None]*23

    '''
    Parameters from optimizer   
      VmaxUp    = param(1)
      g_irel_max  = param(2)
      RyRa1         = param(3)
      RyRa2         = param(4)
      RyRahalf      = param(5)
      RyRohalf      = param(6)
      RyRchalf      = param(7)
      kNaCa         = param(8)
      PNaK          = param(9)
      Kup     = param(10)
      V_leak    = param(11)
      alpha         = param(12)
    '''
    VmaxUp = 0.5113      # millimolar_per_second (in calcium_dynamics)
    g_irel_max = 62.5434 # millimolar_per_second (in calcium_dynamics)
    RyRa1 = 0.05354      # uM
    RyRa2 = 0.0488       # uM
    RyRahalf = 0.02427   # uM
    RyRohalf = 0.01042   # uM
    RyRchalf = 0.00144   # uM
    kNaCa = 3917.0463    # A_per_F (in i_NaCa)
    PNaK = 2.6351        # A_per_F (in i_NaK)
    Kup = 3.1928e-4      # millimolar (in calcium_dynamics)
    V_leak = 4.7279e-4   # per_second (in calcium_dynamics)
    alpha = 2.5371       # dimensionless (in i_NaCa)

    ## Constants
    F = 96485.3415   # coulomb_per_mole (in model_parameters)
    R = 8.314472   # joule_per_mole_kelvin (in model_parameters)
    T = 310   # kelvin (in model_parameters)

    ## Cell geometry
    V_SR = 583.73   # micrometre_cube (in model_parameters)
    Vc   = 8800   # micrometre_cube (in model_parameters)
    Cm   = 9.87109e-11   # farad (in model_parameters)

    ## Extracellular concentrations
    Nao = 151   # millimolar (in model_parameters)
    Ko  = 5.4   # millimolar (in model_parameters)
    Cao = 1.8 #3#5#1.8   # millimolar (in model_parameters)

    ## Intracellular concentrations
    # Naio = 10 mM y(17)
    Ki = 150   # millimolar (in model_parameters)
    # Cai  = 0.0002 mM y(2)
    # caSR = 0.3 mM y(1)

    ## Nernst potential
    E_Na = R*T/F*log(Nao/y(17))
    E_Ca = 0.5*R*T/F*log(Cao/y(2))

    E_K  = R*T/F*log(Ko/Ki)
    PkNa = 0.03   # dimensionless (in electric_potentials)
    E_Ks = R*T/F*log((Ko+PkNa*Nao)/(Ki+PkNa*y(17)))


    ## INa
    g_Na        = 3671.2302   # S_per_F (in i_Na)
    i_Na        = g_Na*y(13)**3*y(11)*y(12)*(y(0)-E_Na)

    h_inf       = 1/sqrt(1+exp((y(0)*1000+72.1)/5.7))
    alpha_h     = 0.057*exp(-(y(0)*1000+80)/6.8)
    beta_h      = 2.7*exp(0.079*y(0)*1000)+3.1*10**5*exp(0.3485*y(0)*1000)
    
    tau_h = conditional(y(0),-0.0385,1.5/((alpha_h+beta_h)*1000),1.5*1.6947/1000)

    dY[11]   = (h_inf-y(11))/tau_h

    j_inf       = 1/sqrt(1+exp((y(0)*1000+72.1)/5.7))
    alpha_j = conditional(y(0),-0.04,(-25428*exp(0.2444*y(0)*1000)-6.948*10**-6*exp(-0.04391*y(0)*1000))*(y(0)*1000+37.78)/(1+exp(0.311*(y(0)*1000+79.23))),0)
    
    beta_j  = conditional(
            y(0),-0.04,
            0.02424*exp(-0.01052*y(0)*1000)/(1+exp(-0.1378*(y(0)*1000+40.14))),
            0.6*exp((0.057)*y(0)*1000)/(1+exp(-0.1*(y(0)*1000+32))),
        )
    
    tau_j       = 7/((alpha_j+beta_j)*1000)
    dY[12]   = (j_inf-y(12))/tau_j

    m_inf       = 1/(1+exp((-y(0)*1000-34.1)/5.9))**(1/3)
    alpha_m     = 1/(1+exp((-y(0)*1000-60)/5))
    beta_m      = 0.1/(1+exp((y(0)*1000+35)/5))+0.1/(1+exp((y(0)*1000-50)/200))
    tau_m       = 1*alpha_m*beta_m/1000
    dY[13]   = (m_inf-y(13))/tau_m

    

    ## INaL
    myCoefTauM  = 1
    tauINaL     = 200 #ms
    GNaLmax     = 2.3*7.5 #(S/F)
    Vh_hLate    = 87.61
    i_NaL       = GNaLmax* y(18)**(3)*y(19)*(y(0)-E_Na)

    m_inf_L     = 1/(1+exp(-(y(0)*1000+42.85)/(5.264)))
    alpha_m_L   = 1/(1+exp((-60-y(0)*1000)/5))
    beta_m_L    = 0.1/(1+exp((y(0)*1000+35)/5))+0.1/(1+exp((y(0)*1000-50)/200))
    tau_m_L     = 1/1000 * myCoefTauM*alpha_m_L*beta_m_L
    dY[18]   = (m_inf_L-y(18))/tau_m_L

    h_inf_L     = 1/(1+exp((y(0)*1000+Vh_hLate)/(7.488)))
    tau_h_L     = 1/1000 * tauINaL
    dY[19]   = (h_inf_L-y(19))/tau_h_L

    ## If
    E_f         = -0.017   # volt (in i_f)
    g_f         = 30.10312   # S_per_F (in i_f)

    i_f         = g_f*y(14)*(y(0)-E_f)
    i_fNa       = 0.42*g_f*y(14)*(y(0)-E_Na)

    Xf_infinity = 1/(1+exp((y(0)*1000+77.85)/5))
    tau_Xf      = 1900/(1+exp((y(0)*1000+15)/10))/1000
    dY[14]   = (Xf_infinity-y(14))/tau_Xf




    ## ICaL
    g_CaL       = 8.635702e-5   # metre_cube_per_F_per_s (in i_CaL)
    i_CaL       = g_CaL*4*y(0)*F**2/(R*T)*(y(2)*exp(2*y(0)*F/(R*T))-0.341*Cao)/(exp(2*y(0)*F/(R*T))-1)*y(4)*y(5)*y(6)*y(7)

    d_infinity  = 1/(1+exp(-(y(0)*1000+9.1)/7))
    alpha_d     = 0.25+1.4/(1+exp((-y(0)*1000-35)/13))
    beta_d      = 1.4/(1+exp((y(0)*1000+5)/5))
    gamma_d     = 1/(1+exp((-y(0)*1000+50)/20))
    tau_d       = (alpha_d*beta_d+gamma_d)*1/1000
    dY[4]    = (d_infinity-y(4))/tau_d

    f1_inf      = 1/(1+exp((y(0)*1000+26)/3))
    constf1 = 1 + 1433*(y(2)-50*1e-6)
    
    tau_f1      = (20+1102.5*exp(-((y(0)*1000+27)**2/15)**2)+200/(1+exp((13-y(0)*1000)/10))+180/(1+exp((30+y(0)*1000)/10)))*constf1/1000
    dY[5]    = (f1_inf-y(5))/tau_f1
    
    f2_inf      = 0.33+0.67/(1+exp((y(0)*1000+32)/4))
    constf2     = 1
    tau_f2      = (600*exp(-(y(0)*1000+25)**2/170)+31/(1+exp((25-y(0)*1000)/10))+16/(1+exp((30+y(0)*1000)/10)))*constf2/1000
    dY[6]    = (f2_inf-y(6))/tau_f2

    alpha_fCa   = 1/(1+(y(2)/0.0006)**8)
    beta_fCa    = 0.1/(1+exp((y(2)-0.0009)/0.0001))
    gamma_fCa   = 0.3/(1+exp((y(2)-0.00075)/0.0008))
    fCa_inf     = (alpha_fCa+beta_fCa+gamma_fCa)/1.3156
    constfCa    = conditional( y(0), -0.06, 1, conditional(fCa_inf,y(7),1,0) )
    
    tau_fCa     = 0.002   # second (in i_CaL_fCa_gate)
    dY[7]    = constfCa*(fCa_inf-y(7))/tau_fCa

    ## Ito
    g_to        = 29.9038   # S_per_F (in i_to)
    i_to        = g_to*(y(0)-E_K)*y(15)*y(16)

    q_inf       = 1/(1+exp((y(0)*1000+53)/13))
    tau_q       = (6.06+39.102/(0.57*exp(-0.08*(y(0)*1000+44))+0.065*exp(0.1*(y(0)*1000+45.93))))/1000
    dY[15]   = (q_inf-y(15))/tau_q

    r_inf       = 1/(1+exp(-(y(0)*1000-22.3)/18.75))
    tau_r       = (2.75352+14.40516/(1.037*exp(0.09*(y(0)*1000+30.61))+0.369*exp(-0.12*(y(0)*1000+23.84))))/1000
    dY[16]   = (r_inf-y(16))/tau_r

    ## IKs
    g_Ks        = 2.041   # S_per_F (in i_Ks)
    i_Ks        = g_Ks*(y(0)-E_Ks)*y(10)**2*(1+0.6/(1+(3.8*0.00001/y(2))**1.4))

    Xs_infinity = 1/(1+exp((-y(0)*1000-20)/16))
    alpha_Xs    = 1100/sqrt(1+exp((-10-y(0)*1000)/6))
    beta_Xs     = 1/(1+exp((-60+y(0)*1000)/20))
    tau_Xs      = 1*alpha_Xs*beta_Xs/1000
    dY[10]   = (Xs_infinity-y(10))/tau_Xs

    ## IKr
    L0           = 0.025   # dimensionless (in i_Kr_Xr1_gate)
    Q            = 2.3   # dimensionless (in i_Kr_Xr1_gate)
    g_Kr         = 29.8667   # S_per_F (in i_Kr)
    i_Kr         = g_Kr*(y(0)-E_K)*y(8)*y(9)*sqrt(Ko/5.4)

    V_half       = 1000*(-R*T/(F*Q)*log((1+Cao/2.6)**4/(L0*(1+Cao/0.58)**4))-0.019)

    Xr1_inf      = 1/(1+exp((V_half-y(0)*1000)/4.9))
    alpha_Xr1    = 450/(1+exp((-45-y(0)*1000)/10))
    beta_Xr1     = 6/(1+exp((30+y(0)*1000)/11.5))
    tau_Xr1      = 1*alpha_Xr1*beta_Xr1/1000
    dY[8]     = (Xr1_inf-y(8))/tau_Xr1

    Xr2_infinity = 1/(1+exp((y(0)*1000+88)/50))
    alpha_Xr2    = 3/(1+exp((-60-y(0)*1000)/20))
    beta_Xr2     = 1.12/(1+exp((-60+y(0)*1000)/20))
    tau_Xr2      = 1*alpha_Xr2*beta_Xr2/1000
    dY[9]    = (Xr2_infinity-y(9))/tau_Xr2
    
    ## IK1
    alpha_K1 = 3.91/(1+exp(0.5942*(y(0)*1000-E_K*1000-200)))
    beta_K1  = (-1.509*exp(0.0002*(y(0)*1000-E_K*1000+100))+exp(0.5886*(y(0)*1000-E_K*1000-10)))/(1+exp(0.4547*(y(0)*1000-E_K*1000)))
    XK1_inf  = alpha_K1/(alpha_K1+beta_K1)
    g_K1     = 28.1492   # S_per_F (in i_K1)
    i_K1     = g_K1*XK1_inf*(y(0)-E_K)*sqrt(Ko/5.4)

    ## INaCa
    KmCa   = 1.38   # millimolar (in i_NaCa)
    KmNai  = 87.5   # millimolar (in i_NaCa)
    Ksat   = 0.1   # dimensionless (in i_NaCa)
    gamma  = 0.35   # dimensionless (in i_NaCa)
    kNaCa1 = kNaCa   # A_per_F (in i_NaCa)
    i_NaCa = kNaCa1*(exp(gamma*y(0)*F/(R*T))*y(17)**3*Cao-exp((gamma-1)*y(0)*F/(R*T))*Nao**3*y(2)*alpha)/((KmNai**3+Nao**3)*(KmCa+Cao)*(1+Ksat*exp((gamma-1)*y(0)*F/(R*T))))

    ## INaK
    Km_K  = 1   # millimolar (in i_NaK)
    Km_Na = 40   # millimolar (in i_NaK)
    PNaK1 = PNaK   # A_per_F (in i_NaK)
    i_NaK = PNaK1*Ko/(Ko+Km_K)*y(17)/(y(17)+Km_Na)/(1+0.1245*exp(-0.1*y(0)*F/(R*T))+0.0353*exp(-y(0)*F/(R*T)))

    ## IpCa
    KPCa  = 0.0005   # millimolar (in i_PCa)
    g_PCa = 0.4125   # A_per_F (in i_PCa)
    i_PCa = g_PCa*y(2)/(y(2)+KPCa)

    ## Background currents
    g_b_Na = 0.95   # S_per_F (in i_b_Na)
    i_b_Na = g_b_Na*(y(0)-E_Na)

    g_b_Ca = 0.727272   # S_per_F (in i_b_Ca)
    i_b_Ca = g_b_Ca*(y(0)-E_Ca)

    ## Sarcoplasmic reticulum
    i_up = VmaxUp/(1+Kup**2/y(2)**2)

    i_leak = (y(1)-y(2))*V_leak

    dY[3] = 0

    # RyR
    RyRSRCass = (1 - 1/(1 +  exp((y(1)-0.3)/0.1)))
    i_rel = g_irel_max*RyRSRCass*y(21)*y(22)*(y(1)-y(2))

    RyRainfss = RyRa1-RyRa2/(1 + exp((1000*y(2)-(RyRahalf))/0.0082))
    RyRtauadapt = 1 #s
    dY[20] = (RyRainfss- y(20))/RyRtauadapt

    RyRoinfss = (1 - 1/(1 +  exp((1000*y(2)-(y(20)+ RyRohalf))/0.003)))
    RyRtauact = conditional(RyRoinfss,y(21),0.1,1)*18.75e-3
    
    dY[21] = (RyRoinfss- y(21))/RyRtauact

    RyRcinfss = (1/(1 + exp((1000*y(2)-(y(20)+RyRchalf))/0.001)))
    
    RyRtauinact = conditional(RyRcinfss,y(22),1,2)*87.5e-3
    
    dY[22] = (RyRcinfss- y(22))/RyRtauinact
    
    ## Ca2+ buffering
    Buf_C       = 0.25   # millimolar (in calcium_dynamics)
    Buf_SR      = 10   # millimolar (in calcium_dynamics)
    Kbuf_C      = 0.001   # millimolar (in calcium_dynamics)
    Kbuf_SR     = 0.3   # millimolar (in calcium_dynamics)
    Cai_bufc    = 1/(1+Buf_C*Kbuf_C/(y(2)+Kbuf_C)**2)
    Ca_SR_bufSR = 1/(1+Buf_SR*Kbuf_SR/(y(1)+Kbuf_SR)**2)

    ## Ionic concentrations
    #Nai
    dY[17] = -Cm*(i_Na+i_NaL+i_b_Na+3*i_NaK+3*i_NaCa+i_fNa)/(F*Vc*1e-18)
    #caSR
    dY[2]  = Cai_bufc*(i_leak-i_up+i_rel-(i_CaL+i_b_Ca+i_PCa-2*i_NaCa)*Cm/(2*Vc*F*1e-18))
    #Cai
    dY[1]  = Ca_SR_bufSR*Vc/V_SR*(i_up-(i_rel+i_leak))

    i_stim = 0
    

    ## Membrane potential
    dY[0] = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_NaK+i_Na+i_NaL+i_NaCa+i_PCa+i_f+i_b_Na+i_b_Ca-i_stim)
    
    return dY

