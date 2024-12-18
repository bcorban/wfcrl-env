from scipy.interpolate import interpn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from farmshadow.fsTypes import vectorD, vector3, map_string_double
from farmshadow import windfarm, environment,  wakemodels, solver, blockagemodels, tools
from copy import copy

from scipy.optimize import minimize
import pandas as pd
import pybobyqa

class simulatorFS:
    def __init__(self, xcoords, ycoords, deficit="supergaussian", deflection="qian", turbulence="qian", windspeed=8.0, winddir=270.0, windturb=0.05):        
        
        self.diameter = 126.0
        self.hubHeight = 90.0
        
        cpct_data = np.genfromtxt('./data/cpct_data.dat')
        
        w  = vectorD(cpct_data[:,0])
        cp = vectorD(cpct_data[:,1])
        # Saturation de ct
        map_ct = cpct_data[:,2]
        map_ct[map_ct>0.999] = 0.999
        ct = vectorD(map_ct)
        
        self.w_carto = cpct_data[:,0]
        self.cp_carto = cpct_data[:,1]
        self.ct_carto = map_ct
        
        self.map_pitch = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                                   5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                                   15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                   24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        self.map_tsr = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                                 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0,
                                 12.5, 13.0, 13.5,14.0, 14.5])
        
        self.map_ct = np.loadtxt("Ct_NREL5MW.txt", dtype='float')
        self.map_cp = np.loadtxt("Cp_NREL5MW.txt", dtype='float') * 0.9377147
        self.map_ct[self.map_ct>0.999] = 0.999
        # self.map_ct[self.map_ct<0.001] = 0.001
        self.map_cp[self.map_cp<0.0] = 0.0
        
        self.layout = np.array([[xcoords[i], ycoords[i], self.hubHeight] for i in range(len(xcoords))])
        print(self.layout)
        # self.layout = np.asarray([[0.*self.diameter, 0.0, self.hubHeight],
        #                      [4.*self.diameter, 0.0, self.hubHeight],                     
        #                      [8.*self.diameter, 0.0, self.hubHeight],
        #                      [0.*self.diameter, 4.*self.diameter, self.hubHeight],
        #                      [4.*self.diameter, 4.*self.diameter, self.hubHeight],
        #                      [8.*self.diameter, 4.*self.diameter, self.hubHeight],
        #                      ])
        self.num_turbine = len(self.layout)
        
        # self.wind = environment.UniformWind(9.0,270.0,0.05,{},"",True)

        self.wind = environment.PowerLawWind(windspeed, winddir, windturb,
                                              {"alphaPL":0.2, "refHeight":self.hubHeight}, "", False)
        

        self.selectModel(deficit=deficit, deflection=deflection, turbulence=turbulence)
        self.block = blockagemodels.NoBlockage({},False)
        self.meandering = wakemodels.NoMeandering(False)
        self.solv = solver.LocalLinearSuperposition(999.0, 999.0, False)
        
        yaw = np.radians(0.)
        shaftTilt, totalTiltAngle = np.radians(0.), np.radians(0.)
        nr = 12
        ntheta = 12
        useRelativeAreas = True
        
        self.farm = windfarm.WindFarm()
        for iNum,turbLoc in enumerate(self.layout):    
            self.farm.addRotor(windfarm.Rotor("turb"+str(iNum),
                                    vector3(turbLoc[0], turbLoc[1], turbLoc[2]),
                                    self.diameter, shaftTilt, yaw, totalTiltAngle, w, cp, ct, nr, ntheta, useRelativeAreas
                                    )
                        )

        # self.farm.SortRotors()
        self.rotors = self.farm.getAlphabeticallySortedRotors() #self.farm.getSortedRotors()
        
        cpCosPower, ctCosPower = 2, 2
        for rotor in self.rotors:
            rotor.setMisalignmentPowerLaw(cpCosPower, ctCosPower)
            
            
        lafun = lambda x:self.interp_map_cp(x[0], x[1], cp_value=self.map_cp.max()+0.1)
        res = minimize(lafun, [10.0,2.0], method="powell",
                       bounds=[(self.map_tsr.min()+0.01,self.map_tsr.max()-0.01),
                               (self.map_pitch.min()+0.01,self.map_pitch.max()-0.01)])
        
        self.tsr_cpmax = res.x[0]
        self.pitch_cpmax = res.x[1]
            
    def set_Cp_i(self, cp_value, i_turb):
        # print(i_turb)
        # print(f'old_Cp : {self.rotors[i_turb].getCp()}')
        # print(f'desired Cp : {cp_value}')
        tsr_pitch, ct_flag = self.get_tsr_pitch_from_map_cp(cp_value)
        ct_new = self.interp_map_ct(tsr_pitch, ct_value=0.0)
            
        cp_carto_new = vectorD(np.ones(len(self.w_carto))*cp_value) 
        ct_carto_new = vectorD(np.ones(len(self.w_carto))*ct_new)
        
        self.rotors[i_turb].setCpCtMap(vectorD(self.w_carto),cp_carto_new,ct_carto_new)    
        # print(f'new_Cp : {self.rotors[i_turb].getCp()}')
        return tsr_pitch[1] , ct_flag

    def get_tsr_pitch_from_map_cp(self, cp_value):

        tsr_target = self.tsr_cpmax        
        lafun = lambda pitch:self.interp_map_cp(tsr_target, pitch[0], cp_value=cp_value)
        # try:
            # res = minimize(lafun, x0=1.0, method='powell',
            #                       bounds=[(0.0,self.map_pitch.max()-0.1)])
        res = pybobyqa.solve(lafun, np.array([1.0]),bounds=([0.0],[self.map_pitch.max()-0.01]))
        # except:
        #     print("ca")
        
        ct_new = self.interp_map_ct(np.array([tsr_target, res.x[0]]), ct_value=0.0)
        ct_flag=False
        if ct_new<0: 
            print("ct<0")
            ct_flag=True
        return [tsr_target,res.x[0]], ct_flag
    
    def interp_map_cp(self, tsr, pitch, cp_value):
        x = np.array([tsr, pitch])
        return abs(interpn((self.map_tsr, self.map_pitch),self.map_cp, x) - cp_value)[0]
    
    def interp_map_ct(self, x, ct_value):
        
        if ct_value == 0.0:
            return interpn((self.map_tsr, self.map_pitch),self.map_ct, x)
        else:
            return abs(interpn((self.map_tsr, self.map_pitch),self.map_ct, x) - ct_value)
        
    def get_Cts(self):
        return np.array([rotor.getCt() for rotor in self.rotors])

    def get_Cps(self):
        return np.array([rotor.getCp() for rotor in self.rotors])
    
    def get_param_df(self):
        
        df = pd.DataFrame()
        df['cps'] = self.get_Cps()
        df['cts'] = self.get_Cts()
        tsr_pitch = []
        for cp_value in self.get_Cps():
            tsr_pitch.append(self.get_tsr_pitch_from_map_cp(cp_value)[0])
        tsr_pitch = np.array(tsr_pitch)
        df['TSR'] = tsr_pitch[:,0]
        df['pitch'] = tsr_pitch[:,1]
        df['power'] = self.computePower()
        df['yaws'] = self.get_yaws()
        return df
    
    def get_pitchs(self):
        tsr_pitch = []
        for cp_value in self.get_Cps():
            tsr_pitch.append(self.get_tsr_pitch_from_map_cp(cp_value)[0])
        tsr_pitch = np.array(tsr_pitch)
        return tsr_pitch[:,1]
        
    def reset_carto(self):        
        for rotor in self.rotors:
            rotor.setCpCtMap(vectorD(self.w_carto),vectorD(self.cp_carto),vectorD(self.ct_carto))
    
    def selectModel(self, deficit="supergaussian", deflection="qian", turbulence="qian"):
        # velocity_model: gaussian, supergaussian, jensen, ishihara, generic
        # deflection_model: no_defl, jimenez, bastankhah, supergaussian, qian
        # turbulence_model: no_turb, tian, qian
        
        if deficit=="supergaussian":
            self.udef = wakemodels.SuperGaussianDeficit({}, False)
        elif deficit=="gaussian":
            self.udef = wakemodels.GaussianDeficit({}, False)
        elif deficit=="jensen":
            self.udef = wakemodels.JensenDeficit({}, False)
        elif deficit=="ishihara":
            self.udef = wakemodels.IshiharaDeficit({}, False)
        elif deficit=="generic":
            self.udef = wakemodels.GenericDeficit({}, False)            
        
        if deflection=="qian":
            self.defl = wakemodels.QianWakeDeflection({}, False)
        elif deflection=="supergaussian":
            self.defl = wakemodels.SuperGaussianWakeDeflection({}, False)
        elif deflection=="no_defl":
            self.defl = wakemodels.NoWakeDeflection({}, False)
        elif deflection=="jimenez":
            self.defl = wakemodels.JimenezWakeDeflection({}, False)
        elif deflection=="bastankhah":
            self.defl = wakemodels.BastankhahWakeDeflection({}, False)
        
        if turbulence=="qian":
            self.wat = wakemodels.QianWakeAddedTurbulence({}, False)
        elif turbulence=="tian":
            self.wat = wakemodels.TianWakeAddedTurbulence({}, False)
        elif turbulence=="no_turb":
            self.wat = wakemodels.NoWakeAddedTurbulence({}, False)
        
        
    def updateFarm(self):            
        self.solv.updateFarm(self.farm, self.wind,
                             self.udef, self.wat,
                             self.defl, self.block, self.meandering)
        
    def computePower(self):        
        power = []
        for rotor in self.rotors:
            power.append(rotor.getPower(self.wind))
        return np.array(power)
    
    def setYawsFarm(self,yaws_deg):
        for rotor,yaw_i in zip(self.rotors,yaws_deg):
            rotor.setYawAngle(float(np.radians(yaw_i)))

    def setYawsi(self,yaw_deg, i_turb):
        rotor=self.rotors[i_turb]
        rotor.setYawAngle(float(np.radians(yaw_deg)))
    
    def get_yaws (self):
        yaws=[]
        for rotor in self.rotors:
            yaws.append(np.rad2deg(rotor.getYawAngle()))
        return np.array(yaws)
    
    def WindAtScatterNodes(self, x, y, z):
        
        xis = vectorD(x)
        yis = vectorD(y)
        zis = vectorD(z)        
        nodes = self.solv.windAtScatterNodesWithRotation(xis, yis, zis, self.farm,
                                             self.wind, self.udef,
                                             self.wat, self.defl,
                                             self.block, self.meandering)
        
        pos, vels, turb = tools.nodesToNumpyVec(self.solv, nodes)
        
        return pos, vels, turb
    
    def compute_lidar(self):
        
        dataLidar_one = pd.read_csv("coordLidar_real_1turbine.csv")
        dataLidar = pd.DataFrame()
        for offset in self.layout:
            df = copy(dataLidar_one)
            df['x'] = df['x'] + offset[0]
            df['y'] = df['y'] + offset[1]
            dataLidar = pd.concat([dataLidar,df])
        
        xis = dataLidar['x'].to_numpy()
        yis = dataLidar['y'].to_numpy()
        zis = dataLidar['z'].to_numpy()
        
        pos, vels, turb = self.WindAtScatterNodes(xis, yis, zis)
        dataLidar['u'] = vels[:, 0]
        return dataLidar
    
    def plotHorPlane(self, xlim=list(), ylim=list(), xres=int(), yres=int(), zHor=None, ulim=list(), fig=None, ax=None, plotType='matplotlib'):
        
        if zHor is None: zHor = self.hubHeight
        
        if not xlim:
            xlim = [min(self.layout[:,0])-1.5*self.diameter,
                    max(self.layout[:,0])+5.5*self.diameter]
        if not ylim:
            ylim = [min(self.layout[:,1])-2.5*self.diameter,
                    max(self.layout[:,1])+2.5*self.diameter]  
        if not xres:
            xres = int((xlim[1]-xlim[0])/10)
        if not yres:
            yres = int((ylim[1]-ylim[0])/10)
            
        x = np.linspace(xlim[0],xlim[1],xres)
        y = np.linspace(ylim[0],ylim[1],yres)

        xis = vectorD(x)
        yis = vectorD(y)
        zis = vectorD([zHor])

        nodes = self.solv.windAtNodes(xis, yis, zis, self.farm, self.wind, self.udef,
                                      self.wat, self.defl, self.block, self.meandering)
        pos, vels, turb = tools.nodesToNumpyVec(self.solv, nodes)
        vectorUx = vels[:, 0]
        ux = np.transpose(np.reshape(vectorUx, (len(xis), len(yis))))

        xx, yy = np.meshgrid(x, y)
        
        if not ulim:
            ulim = [ux.min(),ux.max()]
        
        if fig is None: fig, ax = plt.subplots()
        fig.set_figwidth(12)
        fig.set_figheight(5)
        divider = make_axes_locatable(ax)        
        cf = ax.contourf(xx,yy,ux,np.linspace(ulim[0],ulim[1],100),cmap="coolwarm")
        ax.contour(xx,yy,ux,np.linspace(ulim[0],ulim[1],10),
                      colors='grey',alpha=0.8,linewidths=1)
        ax.axis('scaled')
        plt.axis('scaled')
        
        axCB = divider.append_axes("right", size="3.5%", pad=0.05)
        plt.colorbar(cf, cax=axCB, label='Velocity')
        
        return fig, ax
    
    def fopt(self, yaw_vector):
        self.setYawsFarm(yaw_vector)
        self.updateFarm()
        return -sum(self.computePower())
    
    def maximize_power(self, bounds=(-40,0)):
        return minimize(self.fopt,method="powell",
                                   bounds=[bounds for _ in range(self.num_turbine)],
                                   x0=1+np.ones(self.num_turbine))
    
if __name__ == "__main__":    
    
   
    # First farm configuration
    farm1 = simulatorFS()
    
    farm1.setYawsFarm(np.array([0,0,0,0,0,0]))
    farm1.updateFarm()
    farm1.plotHorPlane(ulim=[0.5,8])
    P_init = sum(farm1.computePower())*1e-6
       
    solSci = farm1.maximize_power()
    P_opti = -solSci.fun*1e-6
    
    df = pd.DataFrame({"P_init" : round(P_init,2), "P_opti" : round(P_opti,2), "Ecart[%]" : round(100*(P_opti-P_init)/P_init,2),              
                  "Yaws":[solSci.x.round(1)]},index=[0])
    
    print(df)
    farm1.plotHorPlane(ulim=[0.5,8])
    plt.show()
    print(f' {farm1.get_param_df()}')
    # Second farm configuration
    farm2 = simulatorFS(deficit="jensen", deflection="jimenez", turbulence="qian")
    
    farm2.setYawsFarm(np.array([0,0,0,0,0,0]))        
    farm2.updateFarm()
    farm2.plotHorPlane(ulim=[0.5,8])
    P_init = sum(farm2.computePower())*1e-6
    
    solSci = farm2.maximize_power()
    
    P_opti = -solSci.fun*1e-6
    
    df = pd.DataFrame({"P_init" : round(P_init,2), "P_opti" : round(P_opti,2), "Ecart[%]" : round(100*(P_opti-P_init)/P_init,2),              
                  "Yaws":[solSci.x.round(1)]},index=[0])
    
    print(df)
    fig, ax = farm2.plotHorPlane(ulim=[0.5,8])
    