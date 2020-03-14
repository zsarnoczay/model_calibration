# -*- coding: utf-8 -*-
#
# Copyright (C) 2018, Adam Zsarnóczay
# 
# This file is part of roka.
# 
# roka is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# roka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with roka.  If not, see <http://www.gnu.org/licenses/>.

"""
roka: A collection of tools for structural performance assessment
=================================================================

Documentation is available in the docstrings and
online at https://github.com

"""
__version__ = '0.0.1'

from roka.basics import *

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.integrate import simps

import openseespy.opensees as ops

from IPython.display import display, HTML

def show_df(df):
    """
    Display a DataFrame with beautiful formatting in Jupyter.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to be displayed. It also works with Series, they are
        automatically converted to DataFrames before displaying them.

    """
    if type(df) is pd.Series:
        display(HTML(df.to_frame().to_html()))
    else:
        display(HTML(df.to_html()))

def get_cumulative_response(response):
    """
    Returns a response with monotonically increasing, cumulative values.
    
    Parameters
    ----------
    response: Series
        The list of response quantities to process. Any arbitrary response is
        acceptable.

    Returns
    -------
    cumulative_response: Series
        The cumulative response quantities.

    """
    # get a list of increments with the first one being the first data point
    # (i.e. the first step is a step from zero to the first data point)
    increments = np.insert(np.diff(response), 0, response[0])
    
    # prepare the list of cumulative responses
    cumulative_response = np.zeros(len(increments))
    for i in range(1, len(increments)):
        cumulative_response[i] = cumulative_response[i-1] + abs(increments[i])
    
    return pd.Series(cumulative_response)

def add_basic_features(response):
    """
    Identifies half-cycles and calculates tangent stiffness and strain energy.
    
    Parameters
    ----------
    response

    Returns
    -------

    """

    response['d_c'] = get_cumulative_response(response['d'])

    # then we identify the half-cycles in the load history
    response['half_cycle'] = np.zeros(len(response.index), dtype=int)
    d_diff_sgn = np.sign(np.diff(response['d'].values))
    ro = 0  # ro - row_offset
    for i, dd in enumerate(d_diff_sgn):
        if dd == 0:
            dd = dd_pre
        if i > 0:
            if dd != dd_pre:
                hc_id = response.loc[i + ro, 'half_cycle'] + 1
                response.loc[i + ro + 1:, 'half_cycle'] = (
                    np.ones(len(response.loc[i + ro + 1:].index),
                            dtype=int) * hc_id)
                first_row = response.loc[i + ro].to_frame().T
                first_row['half_cycle'] = response.loc[i + ro + 1, 'half_cycle']
                response = pd.concat([response.iloc[:i + ro + 1], first_row,
                                      response.iloc[i + ro + 1:]]).reset_index(
                    drop=True)
                ro += 1

        dd_pre = dd

    # store the load directions
    response['load_dir'] = np.insert(np.sign(np.diff(response['d'].values)), 0,
                                     0)
    for i in response.index:
        if response.loc[i, 'load_dir'] == 0:
            response.loc[i, 'load_dir'] = response.loc[i + 1, 'load_dir']

    response = response[response['load_dir'] != 0]
    response.index = np.arange(len(response.index))

    # calculate response characteristics for each half cycle and also cumulative
    response['K'] = np.zeros(len(response.index))
    response['E'] = np.zeros(len(response.index))
    response['E_c'] = np.zeros(len(response.index))
    for c_i in range(response['half_cycle'].values[-1] + 1):
        c_i_list = response.index[response['half_cycle'] == c_i]
        F = response.loc[c_i_list, 'F'].values
        d_c = response.loc[c_i_list, 'd_c'].values
        
        # K stiffness is calculated as the gradient of the F(d_c) function
        # The calculation is based on second-order accurate central differences 
        # in the interior points and first order accurate differences at the boundaries
        response.loc[c_i_list, 'K'] = np.gradient(F, d_c)
        response['K'].fillna(100., inplace=True)  #todo: fix this 100

        # E energy is calculated using numerical integration for each half-cycle
        # Simpson's rule is used to improve accuracy
        response.loc[c_i_list[1:], 'E'] = [
            simps(np.abs(F[:i + 1]), 
                  d_c[:i + 1], 
                  even='first'
                  ) for i in range(1, len(F))]
        response['E'].fillna(0., inplace=True)
        response.loc[c_i_list, 'E_c'] = (response.loc[c_i_list, 'E'] +
                                         response.loc[
                                             max(c_i_list[0] - 1, 0), 'E_c'])
    return response

class Material(object):
    """
    A material model that can be applied to components.
    
    Attributes
    ----------
    eps_y,
    
    id: int
        Unique material id for FEM representation.
    E_0: float
        Material stiffness (i.e. stress over unit strain) in Pa.
    
    
    """
    def __init__(self, id, E_0, f_y):
        self.id = int(id)
        self.E_0 = float(E_0)
        self.f_y = float(f_y)
        
    @property
    def eps_y(self):
        """The yield strain of the material."""
        if self.f_y is not None:
            return self.f_y / self.E_0
        else:
            return None
    
class Elastic(Material):
    """
    A linear elastic uniaxial material object.
    
    """
    def __init__(self, id, E_0):
        """Initialize the Elastic material"""
        Material.__init__(self, id, E_0, None)
        
    def create_FEM(self):
        """
        Create the representation of the elastic material in OpenSEES.

        """
        ops.uniaxialMaterial('Elastic', self.id, self.E_0)

class Steel4(Material):
    """
    A complex nonlinear material object for metals.
    
    Attributes
    ----------    
    non_sym: bool, optional
    iso: bool, optional
    kin: bool, optional
    ult: bool, optional
    
    b_k: float, optional
    R_0: float, optional
    r_1: float, optional
    r_2: float, optional
    
    b_i: float, optional
    rho_i: float, optional
    b_l: float, optional
    R_i: float, optional
    l_yp: float, optional
    
    f_u: float, optional
    R_u: float, optional
    
    b_kc: float, optional
    R_0c: float, optional
    r_1c: float, optional
    r_2c: float, optional
    b_ic: float, optional
    rho_ic: float, optional
    b_lc: float, optional
    R_ic: float, optional
    f_uc: float, optional
    R_uc: float, optional
    
    """
    def __init__(self, id, E_0=210*GPa, f_y=250*MPa,
                 non_sym=False, iso=True, kin=True, ult=True,
                 b_k=0.005, R_0=25., r_1=0.9, r_2=0.15,
                 b_i=0.005, rho_i=0.2, b_l=0.001, R_i=3., l_yp=1.,
                 f_u=410 * MPa, R_u=5.,
                 b_kc=0.005, R_0c=25., r_1c=0.9, r_2c=0.15,
                 b_ic=0.005, rho_ic=0.2, b_lc=0.001, R_ic=3.,
                 f_uc=410*MPa, R_uc=5.,
                 ):
        """Initialize the Steel4 material."""
        Material.__init__(self, id, E_0, f_y)

        self.mat_tags = []
        if non_sym:
            self.mat_tags.append('non_sym')

        if kin:
            self.mat_tags.append('kin')
            self.b_k = float(b_k)
            self.R_0 = float(R_0)
            self.r_1 = float(r_1)
            self.r_2 = float(r_2)

            if non_sym:
                self.b_kc = float(b_kc)
                self.R_0c = float(R_0c)
                self.r_1c = float(r_1c)
                self.r_2c = float(r_2c)
            else:
                self.b_kc = self.b_k
                self.R_0c = self.R_0
                self.r_1c = self.r_1
                self.r_2c = self.r_2
        else:
            self.b_k = 1e-10
            self.R_0 = 200.0
            self.r_1 = 0.05
            self.r_2 = 0.15
            self.b_kc = self.b_k
            self.R_0c = self.R_0
            self.r_1c = self.r_1
            self.r_2c = self.r_2

        if iso:
            self.mat_tags.append('iso')
            self.b_i = float(b_i)
            self.rho_i = float(rho_i)
            self.b_l = float(b_l)
            self.R_i = float(R_i)
            self.l_yp = float(l_yp)

            if non_sym:
                self.b_ic = float(b_ic)
                self.rho_ic = float(rho_ic)
                self.b_lc = float(b_lc)
                self.R_ic = float(R_ic)
            else:
                self.b_ic = self.b_i
                self.rho_ic = self.rho_i
                self.b_lc = self.b_l
                self.R_ic = self.R_i
        else:
            self.b_i = 1e-10
            self.rho_i = 1.0
            self.b_l = 1e-10
            self.R_i = 200.
            self.l_yp = 0.
            self.b_ic = self.b_i
            self.rho_ic = self.rho_i
            self.b_lc = self.b_l
            self.R_ic = self.R_i

        if ult:
            self.mat_tags.append('ult')
            self.f_u = float(f_u)
            self.R_u = float(R_u)

            if non_sym:
                self.f_uc = float(f_uc)
                self.R_uc = float(R_uc)
            else:
                self.f_uc = self.f_u
                self.R_uc = self.R_u
        else:
            self.f_u = 1e8 * self.f_y
            self.R_u = 50.
            self.f_uc = self.f_u
            self.R_uc = self.R_u
    
    def create_FEM(self):
        """Create the representation of the Steel4 material in OpenSEES."""
        ops.uniaxialMaterial('Steel4', self.id, self.f_y, self.E_0,
                             '-asym',
                             '-kin',
                             self.b_k, self.R_0, self.r_1, self.r_2,
                             self.b_kc, self.R_0c, self.r_1c, self.r_2c,
                             '-iso',
                             self.b_i, self.rho_i, self.b_l, self.R_i,
                             self.l_yp,
                             self.b_ic, self.rho_ic, self.b_lc, self.R_ic,
                             '-ult',
                             self.f_u, self.R_u,
                             self.f_uc, self.R_uc)

class Component(object):
    """
    A component that is used as a building block of the structural system.
    
    """
    def __init__(self):
        pass

class Truss(Component):
    """
    A general truss element represented by corotTruss in OpenSEES.
    
    Attributes
    ----------
    material: Material
        Defines the material of the truss element that describes its 
        force-displacement behavior.
    l_tot: float, optional
        The total length of the truss element in m. Default: 1.0.
    A_cs: float, optional
        The cross-section area of the truss element in m2. Default: 1.0.
    """
    
    def __init__(self, material, l_tot=1., A_cs=1.):
        """Initialize the truss element."""
        Component.__init__(self)
        self.material = material
        self.l_tot = l_tot
        self.A_cs = A_cs
        
    def create_FEM(self, component_id, node_i, node_j):
        """
        Create the representation of the truss component in OpenSEES.
        
        Parameters
        ----------
        component_id: int
            The element id assigned to the component in OpenSEES.
        node_i: int
            The id of the start node of the component in OpenSEES.
        node_j: int
            The id of the end node of the component in OpenSEES.
        """
        ops.element('corotTruss', component_id, node_i, node_j, self.A_cs, 
                    self.material.id, '-doRayleigh', 1)
    
class BRB(Truss):
    """
    A special truss element that represents a buckling restrained brace.
    
    Attributes
    ---------- 
    
    id: int
        Unique material id for BRB. (Should be replaced with a more convenient 
        approach.)
    A_y: float
        Cross-section area of the yielding zone of the BRB steel core in m2.
    l_tot: float
        Workpoint-to-workpoint length of the BRB in m.
    f_SM: float
        Stiffness-modification factor.
    f_DM: float
        Deformation-modification factor.
    f_yd: float
        Design yield strength of the BRB steel core.
    gamma_ov: float
        Overstrength factor.
    
    """
    def __init__(self, id, A_y, l_tot, f_SM, f_DM, f_yd, gamma_ov):
        """Initialize the BRB object."""
        # store the parameters
        self.f_SM = float(f_SM)
        self.f_DM = float(f_DM)
        self.f_yd = float(f_yd)
        self.gamma_ov = float(gamma_ov)
        # create the custom Steel4 material for the BRB
        f_A = np.sqrt(A_y/0.005)
        f_y = f_yd * gamma_ov
        BRB_material = Steel4(id, 
                              E_0 = 210.*GPa*self.f_SM, 
                              f_y = f_y,
                              non_sym = True,
                              kin=True,
                              b_k=0.005, 
                              R_0=25., 
                              r_1=0.91, r_2=0.1,
                              b_kc=0.03 - 0.005*f_A, 
                              R_0c=25., 
                              r_1c=0.89, r_2c=0.02,                              
                              iso = True,
                              b_i=0.003 - 0.0005*f_A, 
                              rho_i=0.25 + 0.05*f_A, 
                              b_l=0.0001, 
                              R_i=3., 
                              l_yp=1.,
                              b_ic=0.005 + 0.001*f_A, 
                              rho_ic=0.25 + 0.05*f_A, 
                              #b_lc=0.0004 - 0.00007*f_A,
                              b_lc = 0.0001,
                              R_ic=3.,
                              ult = True,
                              f_u=1.75*f_y, #1.65*f_y,
                              f_uc=2.50*f_y,
                              R_u=2., # 5., 
                              R_uc = 2.) #5.)

        Truss.__init__(self,BRB_material, l_tot, A_y)
        
    @property
    def A_y(self):
        return self.A_cs
    
class Structure(object):
    """
    Numerical representation of a structural system.
    
    Attributes
    ----------
    T, ctrl_node
    
    xi: float, optional
        Damping ratio for Rayleigh damping. Default: 0.05.
    """

    def __init__(self, xi):
        """ Initialize the structural system"""
        self.xi = xi
        self._T = None
        self.ctrl_node = 1

    @property
    def T(self):
        """
        Calculate and return the fundamental vibration period of the system.

        Returns
        -------
        T: float
            The fundamental vibration period in seconds.
        """
        if self._T == None:
            T_list = Analysis().eigen(self, 1)
            self._T = T_list[1]

        return self._T

    
class SDOF(Structure):
    """
    A single-degree-of-freedom system.
    
    Attributes
    ----------
    T,
    
    component: Component
        The structural component that represents the force-displacement behavior
        of the system.
    mass: float, optional
        Lumped mass at the end of the component in kg. Default: 10,000.
    """
    
    def __init__(self, component, mass=1e5, xi=0.05):
        """Initialize the SDOF system."""
        Structure.__init__(self, xi=xi)
        self.component = component
        self.mass = mass   
    
    def create_FEM(self, damping=True):
        """
        Creates the representation of the SDOF system in OpenSEES.

        Parameters
        ----------
        damping: bool
            Controls the assignment of Rayleigh damping. If False, no damping
            is assigned for the system.
        """
        # support nodes
        ops.node(0, 0.)
        ops.fix(0, 1)
        
        # free node
        ops.node(1, self.component.l_tot, '-mass', self.mass)
        
        # material
        self.component.material.create_FEM()
        
        # component
        self.component.create_FEM(component_id=1, node_i=0, node_j=1)
        
        # damping
        if self.xi > 0.001 and damping:
            # force recalculation of the natural period
            self._T = None
            
            # calculate the damping coefficients
            omega_1 = 2. * np.pi / self.T
            omega_2 = 2. * np.pi / (self.T * 16.)
            a0 = self.xi * 2. * omega_1 * omega_2 / (omega_1 + omega_2)
            b0 = self.xi * 2. / (omega_1 + omega_2)
            
            # apply damping to the component
            ops.region(1, '-ele', 1, '-rayleigh', a0, b0, 0., 0.)


class DemandProtocol(object):
    """
    Stores and generates a demand-history for analysis and evaluation.

    Attributes
    ----------
    demand_list: ndarray
        Values that describe the demand history. Linear interpolation is used 
        to get intermediate values.
    """
    def __init__(self, demand_list):
        """Initialize the demand-history object."""
        self.demand_list = demand_list
        self._step_size = None
        self._calc_values()
        
    @property
    def step_size(self):
        """
        A pre-defined step-size that is used to discretize the demand-history.
        The default step_size is None, meaning that the demand-history is
        applied without additional interpolation between points.
        
        """
        return self._step_size
    
    @step_size.setter
    def step_size(self, value):
        self._step_size = value
        self._calc_values()
        
    def _calc_values(self):
        """
        Calculates a list of demand values considering the specified step size.
        
        """
        if self._step_size == None:
            self.values = deepcopy(self.demand_list)
        else:
            values = np.zeros(1)
            for val in self.demand_list:
                val_pre = values[-1]
                if val_pre < val:
                    values = np.concatenate((values, 
                                             np.arange(val_pre, val, 
                                                       self._step_size)[1:]))
                else:
                    values = np.concatenate((values, 
                                             np.arange(val_pre, val, 
                                                       -self._step_size)[1:]))
                if np.abs(val-values[-1]) > self._step_size/1e6:
                    values = np.append(values, val)
            self.values = values[1:]
        self._calc_increments()
        
    def _calc_increments(self):
        """
        Calculates demand increments from the demand values.
        
        """
        self.increments = np.insert(np.diff(self.values),0,self.values[0])
        
    @property
    def length(self):
        """
        Return the number of values in the demand-history.
        """
        return len(self.values)

class Analysis(object):
    """
    Collects settings and performs finite element analysis in OpenSEES.
    
    Attributes
    ----------
    ndm: int, optional
        Number of dimensions. Default: 1.
    ndf: int, optional
        Number of degrees of freedom. Default: 1.
        
    """
    def __init__(self, ndm=1, ndf=1):
        """
        Create the OpenSees model object.
        """
        self.ndm = ndm
        self.ndf = ndf
        
    def _initialize(self):
        ops.wipe()
        ops.model('basic', '-ndm', self.ndm, '-ndf', self.ndf)
        
    def eigen(self, struct, mode_count=1):
        """
        Perform a modal analysis and get the vibration periods of the structure. 
        
        Parameters
        ----------
        struct: Structure
            The structure or system to analyze.
        mode_count: int, optional
            The number of vibration modes to evaluate. Default: 1.

        Returns
        -------
        T_list: Series
            A list of vibration periods corresponding to the requested number of
            vibration modes. 

        """
        # initialize the analysis
        self._initialize()

        # define the structure - remove damping 
        struct = deepcopy(struct)
        struct.xi = 0.
        struct.create_FEM()
        
        eigens = ops.eigen('-fullGenLapack', mode_count)
        T_list = 2 * np.pi / np.sqrt(np.array(eigens))
        
        return pd.Series(T_list, index=np.arange(1,mode_count+1))
        
    def material_response(self, material, demand):
        """
        Calculate the response of a material to a given load history.
        
        Parameters
        ----------
        material: Material
            The material object to analyze.
        demand: DemandProtocol
            The load history the material shall be exposed to.
            
        Returns
        -------
        response: DataFrame
            The DataFrame includes columns for strain (eps) and stress (sig).

        """
        # initialize the analysis
        self._initialize()
        
        # define the structure
        struct = SDOF(Truss(material, l_tot=1., A_cs=1.))
        struct.create_FEM(damping=False)
        id_ctrl_node = struct.ctrl_node
        load_dir = 1
        
        # define the loading
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        if self.ndf == 1 and self.ndm == 1:
            ops.load(id_ctrl_node, 1.)
        
        # configure the analysis
        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('NormDispIncr', 1e-10, 100)
        ops.algorithm('NewtonLineSearch', '-maxIter', 100)
        
        # initialize the arrays for results
        result_size = demand.length
        response = pd.DataFrame(np.zeros((result_size+1, 2)), 
                                columns = ['eps', 'sig'])
        
        # perform the analysis
        for i, disp_incr in enumerate(demand.increments):
            ops.integrator('DisplacementControl', id_ctrl_node, load_dir, disp_incr)
            ops.analysis('Static')
            ops.analyze(1)
            response.loc[i+1, 'eps'] = ops.nodeDisp(id_ctrl_node, load_dir)
            response.loc[i+1, 'sig'] = ops.eleResponse(1, 'axialForce')[0]
            
        return response