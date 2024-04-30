# -*- coding: utf-8 -*-
import numpy as np                                        # arrays and matrix math
import numpy.linalg as linalg  # for linear algebra
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

import pandas as pd                                       # DataFrames
import matplotlib.pyplot as plt                           # plotting
import math  # for trig functions etc.
from PIL import Image
import os
import matplotlib.image as mp

dpi = 1000
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 设置字号为10pt，大于7pt
#exe_path = r'K:/JinCode/'
'''
Set the workfile path
'''
exe_path = r'C:/Users/Administrator1/Desktop/gradual deformation/submit/code/'

def Random_path_GD(nxy, seed):
    '''
    make a random array and its numbers size from 0 to nxy
    parm nxy: int, all cells
    parm seed: int, for random path
    '''
    np.random.seed(seed)
    path = np.zeros(nxy)
    ind = 0
    for ixy in range(0,nxy): 
        path[ixy] = ind  
        ind = ind + 1        
    a = np.random.rand(nxy)
    inds = a.argsort()
    path = path[inds]
    return path


def gauinv(p):
    """Compute the inverse of the standard normal cumulative distribution function. 
    :param p: cumulative probability value    :return: TODO
    """
    lim = 1.0e-10
    p0 = -0.322_232_431_088
    p1 = -1.0
    p2 = -0.342_242_088_547
    p3 = -0.020_423_121_024_5
    p4 = -0.000_045_364_221_014_8
    q0 = 0.099_348_462_606_0
    q1 = 0.588_581_570_495
    q2 = 0.531_103_462_366
    q3 = 0.103_537_752_850
    q4 = 0.003_856_070_063_4

    # Check for an error situation
    if p < lim:
        xp = -1.0e10
        return xp
    if p > (1.0 - lim):
        xp = 1.0e10
        return xp

    # Get k for an error situation
    pp = p
    if p > 0.5:
        pp = 1 - pp
    xp = 0.0
    if p == 0.5:
        return xp

    # Approximate the function
    y = np.sqrt(np.log(1.0 / (pp * pp)))
    xp = float(
        y
        + ((((y * p4 + p3) * y + p2) * y + p1) * y + p0)
        / ((((y * q4 + q3) * y + q2) * y + q1) * y + q0)
    )
    if float(p) == float(pp):
        xp = -xp
    return xp

def gcum(x):
    """Calculate the cumulative probability of the standard normal distribution.
    :param x: the value from the standard normal distribution
    :return: TODO
    """

    z = x
    if z < 0:
        z = -z
    t  = 1./(1.+ 0.2316419*z)
    gcum = t*(0.31938153   + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    e2   = 0.
    #  6 standard deviations out gets treated as infinity:
    if z <= 6.:
        e2 = np.exp(-z*z/2.)*0.3989422803
    gcum = 1.0- e2 * gcum
    if x >= 0:
        return gcum
    gcum = 1.0 - gcum
    return gcum

class Observe:
    def locpix_st(array,xmin,xmax,ymin,ymax,step,vmin,vmax,df,xcol,ycol,vcol,title,xlabel,ylabel,vlabel,cmap):
        """
        :param array: ndarray
        :param xmin: x axis minimum
        :param xmax: x axis maximum
        :param ymin: y axis minimum
        :param ymax: y axis maximum
        :param step: step
        :param vmin: TODO
        :param vmax: TODO
        :param df: dataframe
        :param xcol: data for x axis
        :param ycol: data for y axis
        :param vcol: color, sequence, or sequence of color
        :param title: title
        :param xlabel: label for x axis
        :param ylabel: label for y axis
        :param vlabel: TODO
        :param cmap: colormap
        :return: QuadContourSet
        """
        xx, yy = np.meshgrid(
            np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
        )
        cs = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)
        plt.scatter(
            df[xcol],
            df[ycol],
            s=None,
            c=df[vcol],
            marker=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            linewidths=0.8,
            edgecolors="black",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        cbar = plt.colorbar(orientation="vertical")
        cbar.set_label(vlabel, rotation=270, labelpad=20)
       
        
        return cs

    def pixelplt_GD(array, xmin, xmax, ymin, ymax, step,vmin,vmax,
                 title, xlabel, ylabel, vlabel, cmap, fig_name, seed):
        plt.figure(figsize=(8, 6))
        im = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar = plt.colorbar(
            im, orientation="vertical", ticks=np.linspace(vmin, vmax, 10))
        cbar.set_label(vlabel, rotation=270, labelpad=20)
        figure_name = 'pixelplt_'+str(seed)+'.png'
        plt.savefig(figure_name,  dpi=dpi)
        plt.show()  
        
        return


class Save:
    def GSLIB2Dataframe(data_file):
        """Convert GSLIB Geo-EAS files to a pandas DataFrame for use with Python
        methods.

        :param data_file: dataframe
        :return: None
        """

        columns = []
        with open(data_file) as f:
            head = [next(f) for _ in range(2)]  # read first two lines
            line2 = head[1].split()
            ncol = int(line2[0])  # get the number of columns

            for icol in range(ncol):  # read over the column names
                head = next(f)
                columns.append(head.split()[0])

            data = np.loadtxt(f, skiprows=0)
            df = pd.DataFrame(data)
            df.columns = columns
            return df   
    def ndarray2GSLIB(array, data_file):
        """Convert 1D or 2D numpy ndarray to a GSLIB Geo-EAS file for use with
        GSLIB methods.

        :param array: input array
        :param data_file: file name
        :param col_name: column name
        :return: None
        """

        if array.ndim not in [1, 2]:
            raise ValueError("must use a 2D array")

        with open(data_file, "w") as f:
            #f.write(data_file + "\n")
            #f.write("1 \n")
            #f.write(col_name + "\n")

            if array.ndim == 2:
                ny, nx = array.shape

                for iy in range(ny):
                    for ix in range(nx):
                        f.write(str(array[ny - 1 - iy, ix]) + "\n")

            elif array.ndim == 1:
                nx = len(array)
                for ix in range(0, nx):
                    f.write(str(array[ix]) + "\n")
    def GSLIB2ndarray(data_file, kcol, nx, ny):
        """Convert GSLIB Geo-EAS file to a 1D or 2D numpy ndarray for use with
        Python methods

        :param data_file: file name
        :param kcol: TODO
        :param nx: shape along x dimension
        :param ny: shape along y dimension
        :return: ndarray, column name
        """
        if ny > 1:
            array = np.ndarray(shape=(ny, nx), dtype=float, order="F")
        else:
            array = np.zeros(nx)

        with open(data_file) as f:
            head = [next(f) for _ in range(2)]  # read first two lines
            line2 = head[1].split()
            ncol = int(line2[0])  # get the number of columns

            for icol in range(ncol):  # read over the column names
                head = next(f)
                if icol == kcol:
                    col_name = head.split()[0]
            if ny > 1:
                for iy in range(ny):
                    for ix in range(0, nx):
                        head = next(f)
                        array[ny - 1 - iy][ix] = head.split()[kcol]
            else:
                for ix in range(nx):
                    head = next(f)
                    array[ix] = head.split()[kcol]
        return array, col_name
    def Dataframe2GSLIB(data_file, df):
        ncol = len(df.columns)
        nrow = len(df.index)
        with open(data_file, "w") as f:
            f.write(data_file + "\n")
            f.write(str(ncol) + "\n")
            for icol in range(ncol):
                f.write(df.columns[icol] + "\n")
            for irow in range(nrow):
                for icol in range(ncol):
                    f.write(str(df.iloc[irow, icol]) + " ")
                f.write("\n")

    def GdOut2ndarray(filepath):
        #return List<float>, sis_out:simulated value, rand_out:random value, order_out:path
        sis_out,rand_out,order_out = [],[],[]
        f = open(filepath, "r")
        lines = f.readlines()
        f.close()           
        for i in range(len(lines)):
            lineList = lines[i].split(" ")    
            data = []
            for i in range(len(lineList)):
                if lineList[i] != '':                              
                    data.append(float(lineList[i]))            
            sis_out.append(data[0])
            rand_out.append(data[1])
            order_out.append(data[2])
        return sis_out,rand_out,order_out
    
    
    
class Optimization:
    def getDiffBetween(fun1,fun2):
        dif_fun = []
        dif_res = 0
        for i in range(len(fun1)):
            res = abs(fun1[i] - fun2[i])
            dif_fun.append(res)
            dif_res = dif_res + res
        return dif_fun, dif_res 

def addInv2Out(seed,gauniv_vec):
    file_path = exe_path + str(seed) +'gdf.out'
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            lines[i] = lines[i][:-2]
            lines[i] = lines[i] + '  ' + str(float(gauniv_vec[i])) + '\n'
            f.write(lines[i])
        f.close()
            
def saveInv(seed,gauniv_vec):
    file_path = exe_path + str(seed) +'_gauinv.out'
    with open(file_path,"w") as f:
        f.write('SISIM SIMULATIONS:'+'\n')
        f.write(str(1) + '\n')
        f.write('gd Value'+'\n')
        for i in range(len(gauniv_vec)):
            lines = '  ' + str(float(gauniv_vec[i])) + '\n'
            f.write(lines)
        f.close()
            
def setFac(fac,filepath):
    #only for nvar = 1 and categorical number
    with open(filepath,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            n = float(lines[i].strip())
            #print(lines[i])
            if n != fac:
                lines[i] = '0\n'
            else:
                lines[i] = '1\n'
            f.write(lines[i])
        f.close()

def do_sisim_gd(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed):
    #save sis results to local file named currentseed.out 
    #currentseed.out have three columns,each represents its simulated value, random value and order
    filename = changeSisGdPar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed)
    os.system('sisim_gd.exe sisim_gd.par')#run sisim to get a result,this result location cant control,only current file
    filepath = exe_path + filename
    sis_out,rand_out,order_out = Save.GdOut2ndarray(filepath)
    return sis_out,rand_out,order_out

def do_sis_gdf(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    '''
    return a new realization depend on inputting a uniform random numbers vector and a random order vector 
    First, you have to change sis_inv.par, espically the seed in this file
    Second, you have to change sis_inv_cond.out, espically the random number and order
    
    :param threads: phase type
    :param gcdf: global distribution
    :param nx: x number of model
    :param xsiz: cell size
    :param ny: y number of model
    :param ysiz: cell size
    :param nz: z number of model
    :param zsiz: cell size
    :param seed: random seed
    :param order: random path
    
    Return: random path, target function
    '''
    changePar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,'sisim_gdf.par')    
    set_gdf_cond(order,'sisim_gdf_cond.out')
    os.system('sisim_gdf.exe sisim_gdf.par')#run sisim to get a result
    filepath = exe_path + str(seed) +'gdf.out'#for avoiding the production of too many useless file
    sis_out,rand_out,order_out = Save.GdOut2ndarray(filepath)    
    return sis_out,rand_out

def do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order):
    #return a new realization depend on inputting a uniform random numbers vector and a random order vector 
    #First, you have to change sis_inv.par, espically the seed in this file
    #Second, you have to change sis_inv_cond.out, espically the random number and order
    changeSisInvPar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed)    
    set_inv_cond(rand,order)
    os.system('sisim_inv.exe sisim_inv.par')#run sisim to get a result
    filepath = exe_path + 'sis_inv.out'#for avoiding the production of too many useless file
    sis_out,rand_out,order_out = Save.GdOut2ndarray(filepath)    
    return sis_out,rand_out

def changeSisGdPar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed):        
    old_numberthreads = r"-number categories"
    new_numberthreads = str(len(threads)) + '  ' + old_numberthreads + '\n'
    
    old_threads = r"-   thresholds / categories"
    new_threads = ""
    for i in range(len(threads)):
        new_threads = new_threads + str(threads[i]) + '  '
    new_threads = new_threads + '  ' + old_threads + '\n'
    
    old_gcdf = r"-   global cdf / pdf"
    new_gcdf = ""
    for i in range(len(gcdf)):
        new_gcdf = new_gcdf + str(gcdf[i]) + '  '
    new_gcdf = new_gcdf + '  ' + old_gcdf + '\n'
    
    old_x = r"-nx,xmn,xsiz"
    xmn = 0.5 * xsiz
    new_x = str(nx) + '  '  + str(xmn)+ '  ' + str(xsiz)  + '  ' + old_x  + '\n'
    
    old_y = r"-ny,ymn,ysiz"
    ymn = 0.5 * ysiz
    new_y = str(ny) + '  '  + str(ymn)+ '  ' + str(ysiz)  + '  ' + old_y + '\n'
    
    old_z = r"-nz,zmn,zsiz"
    zmn = 0.5 * zsiz
    new_z = str(nz) + '  '  + str(zmn)+ '  ' + str(zsiz)  + '  ' + old_z + '\n'
    
    old_seed = r"-random number seed"
    new_seed = str(seed) + '  ' + old_seed + '\n'
    
    old_filename = r"-file for simulation output"
    new_filename = str(seed) + 'gd.out       ' + old_filename + '\n'
    
    file_path = exe_path + "sisim_gd.par"
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            #print(lines[i])
            if old_numberthreads in lines[i]:
                lines[i] = new_numberthreads
            if old_threads in lines[i]:
                lines[i] = new_threads
            if old_gcdf in lines[i]:
                lines[i] = new_gcdf
            if old_seed in lines[i]:
                lines[i] = new_seed
            if old_x in lines[i]:
                lines[i] = new_x
            if old_y in lines[i]:
                lines[i] = new_y
            if old_z in lines[i]:
                lines[i] = new_z
            if old_filename in lines[i]:
                lines[i] = new_filename
            f.write(lines[i])
        f.close()
    filename = str(seed) + 'gd.out'
    return filename    
 
def changeSisInvPar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed):        
    
    old_numberthreads = r"-number categories"
    new_numberthreads = str(len(threads)) + '  ' + old_numberthreads + '\n'
    
    old_threads = r"-   thresholds / categories"
    new_threads = ""
    for i in range(len(threads)):
        new_threads = new_threads + str(threads[i]) + '  '
    new_threads = new_threads + '  ' + old_threads + '\n'
    
    old_gcdf = r"-   global cdf / pdf"
    new_gcdf = ""
    for i in range(len(gcdf)):
        new_gcdf = new_gcdf + str(gcdf[i]) + '  '
    new_gcdf = new_gcdf + '  ' + old_gcdf + '\n'
    
    old_x = r"-nx,xmn,xsiz"
    xmn = 0.5 * xsiz
    new_x = str(nx) + '  '  + str(xmn)+ '  ' + str(xsiz)  + '  ' + old_x  + '\n'
    
    old_y = r"-ny,ymn,ysiz"
    ymn = 0.5 * ysiz
    new_y = str(ny) + '  '  + str(ymn)+ '  ' + str(ysiz)  + '  ' + old_y + '\n'
    
    old_z = r"-nz,zmn,zsiz"
    zmn = 0.5 * zsiz
    new_z = str(nz) + '  '  + str(zmn)+ '  ' + str(zsiz)  + '  ' + old_z + '\n'
    
    old_seed = r"-random number seed"
    new_seed = str(seed) + '  ' + old_seed + '\n'
    
    file_path = exe_path + "sisim_inv.par"
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            #print(lines[i])
            if old_numberthreads in lines[i]:
                lines[i] = new_numberthreads
            if old_threads in lines[i]:
                lines[i] = new_threads
            if old_gcdf in lines[i]:
                lines[i] = new_gcdf
            if old_seed in lines[i]:
                lines[i] = new_seed
            if old_x in lines[i]:
                lines[i] = new_x
            if old_y in lines[i]:
                lines[i] = new_y
            if old_z in lines[i]:
                lines[i] = new_z
            f.write(lines[i])
        f.close() 
        
def changePar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,filename):        
    old_numberthreads = r"-number categories"
    new_numberthreads = str(len(threads)) + '  ' + old_numberthreads + '\n'
    
    old_threads = r"-   thresholds / categories"
    new_threads = ""
    for i in range(len(threads)):
        new_threads = new_threads + str(threads[i]) + '  '
    new_threads = new_threads + '  ' + old_threads + '\n'
    
    old_gcdf = r"-   global cdf / pdf"
    new_gcdf = ""
    for i in range(len(gcdf)):
        new_gcdf = new_gcdf + str(gcdf[i]) + '  '
    new_gcdf = new_gcdf + '  ' + old_gcdf + '\n'
    
    old_x = r"-nx,xmn,xsiz"
    xmn = 0.5 * xsiz
    new_x = str(nx) + '  '  + str(xmn)+ '  ' + str(xsiz)  + '  ' + old_x  + '\n'
    
    old_y = r"-ny,ymn,ysiz"
    ymn = 0.5 * ysiz
    new_y = str(ny) + '  '  + str(ymn)+ '  ' + str(ysiz)  + '  ' + old_y + '\n'
    
    old_z = r"-nz,zmn,zsiz"
    zmn = 0.5 * zsiz
    new_z = str(nz) + '  '  + str(zmn)+ '  ' + str(zsiz)  + '  ' + old_z + '\n'
    
    old_seed = r"-random number seed"
    new_seed = str(seed) + '  ' + old_seed + '\n'
    
    old_filename = r"-file for simulation output"
    new_filename = str(seed) + 'gdf.out       ' + old_filename + '\n'
    
    file_path = exe_path + filename
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            #print(lines[i])
            if old_numberthreads in lines[i]:
                lines[i] = new_numberthreads
            if old_threads in lines[i]:
                lines[i] = new_threads
            if old_gcdf in lines[i]:
                lines[i] = new_gcdf
            if old_seed in lines[i]:
                lines[i] = new_seed
            if old_x in lines[i]:
                lines[i] = new_x
            if old_y in lines[i]:
                lines[i] = new_y
            if old_z in lines[i]:
                lines[i] = new_z
            if old_filename in lines[i]:
                lines[i] = new_filename
            f.write(lines[i])
        f.close()    
        
def set_inv_cond(rand_out,order):
    filepath = exe_path + 'sisim_inv_cond.out'
    with open(filepath,"r+") as f:
        f.seek(0)
        f.truncate()
        for i in range(len(rand_out)):
            lines = '0  ' + str(float(rand_out[i])) + '  ' + str(order[i]) + '\n'
            f.write(lines)
        f.close()

def set_gdf_cond(order,filename):
    filepath = exe_path + filename
    with open(filepath,"r+") as f:
        f.seek(0)
        f.truncate()
        for i in range(len(order)):
            lines = '0 0 ' + str(order[i]) + '\n'
            f.write(lines)
        f.close()
        
def changeSisPar(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed):        
    old_numberthreads = r"-number thresholds/categories"
    new_numberthreads = str(len(threads)) + '  ' + old_numberthreads + '\n'
    
    old_threads = r"-   thresholds / categories"
    new_threads = ""
    for i in range(len(threads)):
        new_threads = new_threads + str(threads[i]) + '  '
    new_threads = new_threads + '  ' + old_threads + '\n'
    
    old_gcdf = r"-   global cdf / pdf"
    new_gacd = ""
    for i in range(len(gcdf)):
        new_gacd = new_gacd + str(gcdf[i]) + '  '
    new_gacd = new_gacd + '  ' + old_gcdf + '\n'
    
    old_x = r"-nx,xmn,xsiz"
    xmn = 0.5 * xsiz
    new_x = str(nx) + '  '  + str(xmn)+ '  ' + str(xsiz)  + '  ' + old_x  + '\n'
    
    old_y = r"-ny,ymn,ysiz"
    ymn = 0.5 * ysiz
    new_y = str(ny) + '  '  + str(ymn)+ '  ' + str(ysiz)  + '  ' + old_y + '\n'
    
    old_z = r"-nz,zmn,zsiz"
    zmn = 0.5 * zsiz
    new_z = str(nz) + '  '  + str(zmn)+ '  ' + str(zsiz)  + '  ' + old_z + '\n'
    
    old_seed = r"-random number seed"
    new_seed = str(seed) + '  ' + old_seed + '\n'
    
    old_filename = r"-file for simulation output"
    new_filename = str(seed) + '.out       ' + old_filename + '\n'
    
    file_path = exe_path + "sisim.par"
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):
            #print(lines[i])
            if old_numberthreads in lines[i]:
                lines[i] = new_numberthreads
            if old_threads in lines[i]:
                lines[i] = new_threads
            if old_gcdf in lines[i]:
                lines[i] = new_gacd
            if old_seed in lines[i]:
                lines[i] = new_seed
            if old_x in lines[i]:
                lines[i] = new_x
            if old_y in lines[i]:
                lines[i] = new_y
            if old_z in lines[i]:
                lines[i] = new_z
            if old_filename in lines[i]:
                lines[i] = new_filename
            f.write(lines[i])
        f.close()

            
def changePHASE3(connect,filename,nx,ny,nz,dx,dy,dz,nlag):
    #filename without the file type
    with open("PHASE3.PAR", "w") as f:
        f.seek(0)
        f.truncate()#clear all
        f.write(str(1) + "\n")
        f.write(str(connect) + "\n")
        f.write(filename + ".out\n")#data_file limits in 8 bytes
        f.write(str(nx) + " " + str(ny) + " " + str(nz) + "\n")
        f.write(str(dx) + " " + str(dy) + " " + str(dz) + "\n")
        f.write(str(nlag) + "\n")
        f.write(filename+'.STA\n')
        f.write(filename+'.CCO\n')
        f.write(filename+'.COF\n')   
        f.close()
        
def delFirstILines(filename,i):
    #del first i lines
    with open(exe_path + filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[i:])
        f.close()

def splitData(line):
    data = []
    lineList = line.split(" ")
    for i in range(len(lineList)):
        if lineList[i] != '':                              
            data.append(float(lineList[i]))            
    return data[0],data[1]

def getDataInSAT(filepath, nlag):
    xloc = []
    xFun = []
    yloc = []
    yFun = []
    zloc = []
    zFun = []
    xyzAvgLoc = []
    xyzAvgFun = []
    # Open file
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):        
        if '(1,0,0)' in lines[i]:
            i = i + 1
            for j in range(nlag):
                Line = lines[i].strip()
                x,y = splitData(Line)
                xloc.append(x);xFun.append(y)
                i = i + 1
        if '(0,1,0)' in lines[i]:
            i = i + 1
            for j in range(nlag):
                Line = lines[i].strip()
                x,y = splitData(Line)
                yloc.append(x);yFun.append(y)
                i = i + 1
        if '(0,0,1)' in lines[i]:
            i = i + 1
            for j in range(nlag):
                Line = lines[i].strip()
                x,y = splitData(Line)
                zloc.append(x);zFun.append(y)
                i = i + 1
        if 'AVERAGE ALONG X, Y AND Z' in lines[i]:
            i = i + 1
            for j in range(nlag):
                Line = lines[i].strip()
                x,y = splitData(Line)
                xyzAvgLoc.append(x);xyzAvgFun.append(y)
                i = i + 1
        i = i + 1
    return xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun

def do_gauinv(rand_out):
    '''
    Transform rand_out into Gaussian vector
    '''
    rand_inv = np.zeros((len(rand_out),1))
    for i in range(len(rand_out)):
        rand_inv[i] = gauinv(rand_out[i])
    return rand_inv
def do_inv2gcum(rand_inv):
    '''
    Transform rand_inv into Uniform vector
    '''
    rand_out = np.zeros((len(rand_inv),1))
    for i in range(len(rand_inv)):
        rand_out[i] = gcum(rand_inv[i])
    return rand_out

def do_deformation(rand_inv1,rand_inv2,N):
    #N are sample times for t 
    #t = np.random.uniform(-1,1,N) * math.pi
    #t.sort()
    t = np.linspace(-math.pi, math.pi, N)
    rand_inv = []
    for i in range(len(t)):
        inv_arr = rand_inv1*math.cos(t[i]) + rand_inv2*math.sin(t[i])
        rand_inv.append(inv_arr)
    return rand_inv,t
    
def do_connect(fac, sis_out_i,filename,connect,nx,ny,nz, xsiz,ysiz,zsiz, nlag):    
    setFacAndSis_1(fac,sis_out_i) #edit conGd.out ,target facie is 1 and other is 0
    changePHASE3(connect=connect, filename=filename, nx=nx, ny=ny, nz=nz, dx=xsiz, dy=ysiz, dz=zsiz, nlag=nlag) 
    os.system('connect3d_2023')  
    filepath = exe_path + filename + '.STA'
    xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = getDataInSAT(filepath,nlag)
    return xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun

def setFacAndSis_1(fac,sis_out):
    filepath = exe_path + 'conGd.out'
    with open(filepath,"r+") as f:
        f.seek(0)
        f.truncate()
        for i in range(len(sis_out)):
            if sis_out[i] != fac:
                line = '0\n'
            else:
                line = '1\n'
            f.write(line)
        f.close() 
def do_plot_compareSTA(target_fun,fun,nlag,filename,labelname):
    '''
    plot the cpmparsion of corresponding function and target function
    :param target_fun: target function
    :param fun: curve function
    :param nlag: max distance of lag
    :param filename: filename
    
    '''
    plt.plot(range(1,nlag+1), target_fun, 'b^--', alpha=0.6, linewidth=1,label='target')
    plt.plot(range(1,nlag+1), fun, 'go-', alpha=0.7, linewidth=1,label=labelname)
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('Connectivity function')
    min2 = 0.0
    sort_fun = np.sort(target_fun)
    for i in range(len(sort_fun)):
        if sort_fun[i] != 0:
            min2 = sort_fun[i]
            break
    plt.xticks(np.linspace(0,nlag,11))
    #plt.ylim((min2-0.01, 1))
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=dpi,bbox_inches = 'tight')
    plt.show()

def do_plot_cdf(rand_out,nbins,figure_name):
    '''
    Plot the cumulative distribution function
    '''
    #rand_out[i] limited in (0,1)
    plt.figure(figsize=(8,6))
    plt.hist(rand_out, nbins, facecolor='blue', edgecolor='b', alpha=0.5)
    figure_path = exe_path + 'img/' + figure_name
    plt.savefig(figure_path, dpi=dpi)
    plt.show()

def do_plot_objective_function(eval_vec,tracks,scores):
    '''
    plot the objective function of all points
    '''
    plt.figure(figsize=(8, 6))
    tt = np.array(tracks)+1
    plt.plot(range(0,tt[len(tt)-1]),eval_vec[:tt[len(tt)-1]],'-', alpha=0.6, linewidth=1)
    plt.scatter(tt, scores, color='red', s=50)
    plt.plot(tt,scores,'ro--', alpha=1, linewidth=1)
    plt.xlabel('Iterative times')
    plt.ylabel('Objective Function')
    plt.show()

def do_plot_xyz(nlag,fun,seed): 
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,nlag+1), fun[0], 'b^--', alpha=0.6, linewidth=1,label='x')
    plt.plot(range(1,nlag+1), fun[1], 'go-', alpha=0.7, linewidth=1,label='y')
    plt.plot(range(1,nlag+1), fun[2], '*', alpha=0.5, linewidth=1,label='z')
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('Connectivity function')
    min2 = 0.0
    sort_fun = np.sort(fun[0])
    for i in range(len(sort_fun)):
        if sort_fun[i] != 0:
            min2 = sort_fun[i]
            break
        
    plt.xticks(np.linspace(0,nlag,11))
    plt.ylim((min2-0.01, 1))
    figure_path = exe_path + 'img/' + 'xyz'+str(seed)+'.png'
    plt.savefig(figure_path, dpi=dpi)
    plt.show()

def do_plot_xyzAvg(nlag,fun,seed): 
    '''
    plot connectivity function
    :param nlag: max distance of lag
    :param fun: curve function
    :param seed: random seed
    
    '''
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,nlag+1), fun, 'b^-', alpha=0.6, linewidth=1,label='Average XYZ')
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('Connectivity function')
    min2 = 0.0
    sort_fun = np.sort(fun)
    for i in range(len(sort_fun)):
        if sort_fun[i] != 0:
            min2 = sort_fun[i]
            break
    plt.ylim((min2-0.01, 1))
    #plt.yticks(np.linspace(min2-0.005,1,11))
    plt.xticks(np.linspace(0,nlag,11))
    figure_path = exe_path + 'img/' + 'xyzAvg'+str(seed)+'.png'
    plt.savefig(figure_path, dpi=dpi)
    plt.show()

def do_plot_xyz(nlag,fun1,fun2,fun3): 
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,nlag+1), fun1, 'bo-', alpha=0.6, linewidth=1,label='X')
    plt.plot(range(1,nlag+1), fun2, 'kx--', alpha=0.6, linewidth=1,label='Y')
    plt.plot(range(1,nlag+1), fun3, 'g^--', alpha=0.6, linewidth=1,label='Z')
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('Connectivity function')
    min2 = 0.0
    sort_fun = np.sort(fun3)
    for i in range(len(sort_fun)):
        if sort_fun[i] != 0:
            min2 = sort_fun[i]
            break
    plt.ylim((min2-0.01, 1))
    #plt.yticks(np.linspace(min2-0.005,1,11))
    plt.xticks(np.linspace(0,nlag,10))
    plt.show()
    

def plot_vec(x_vec,y_vec,filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x_vec, y_vec, 'b^-', alpha=0.6, linewidth=1,label='Results')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('results')    
    plt.xlim((-math.pi,math.pi))
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=dpi)
    plt.show()
    
def plot_vertical(target_vec,filename):
    plt.figure(figsize=(8, 6))
    depth = np.linspace(1, len(target_vec),len(target_vec))
    plt.plot(target_vec,depth, '.-', alpha=0.6, linewidth=1,label='Results')
    plt.legend()
    plt.xlabel('Total Connected Cells')
    plt.ylabel('Depth')   
    plt.yticks(depth)
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=dpi)
    plt.show()

def do_plot_verticalCompare(target_vec,vertical,filename):
    plt.figure(figsize=(8, 6))
    x = np.linspace(1, len(target_vec),len(target_vec))
    plt.plot(target_vec,x, 'b^--', alpha=0.6, linewidth=1,label='Target cells')
    plt.plot(vertical,x, 'go-', alpha=0.7, linewidth=1,label='Current cells')
    plt.legend()
    plt.ylabel('Depth')
    plt.xlabel('Total Connected Cells')
    plt.yticks(x)
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=dpi)
    plt.show()    
#%%version 2. Objective function using connectivity function
def f(t, target,inv1,inv2, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order,fac,connect,nlag):
    '''
    Do gradual deformation
    :param t: deformation parameter
    :param target: target function
    :param inv1: Gaussian distributed vector
    :param inv2: Gaussian distributed vector
    :param threads: phase type
    :param gcdf: global distribution
    :param nx: x number of model
    :param xsiz: cell size
    :param ny: y number of model
    :param ysiz: cell size
    :param nz: z number of model
    :param zsiz: cell size
    :param seed: random seed
    :param order: random path
    
    Return: 
        dif_res: objective function
        xyzAvgFun: corresponding function
        sis_out_i: SIS Model
        rand_out_i: Random Numbers
    '''
    gd = inv1*math.cos(t) + inv2*math.sin(t)        
    #do_plot_cdf(gd, 1000, 'gauinv gd.png') 
    rand = do_inv2gcum(gd)
    #do_plot_cdf(rand, 1000, 'rand gd.png')     
    sis_out_i,rand_out_i = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order)
    filename = 'conGd'#the file name without file type(.out)
    xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = do_connect(fac,sis_out_i,filename,connect,nx,ny,nz,xsiz,ysiz,zsiz, nlag)
    dif_fun, dif_res = Optimization.getDiffBetween(target, xyzAvgFun)
    return dif_res,xyzAvgFun,sis_out_i,rand_out_i

def results_plot(rand_good_out,avgFun_good, target,nlag,nbins,current_seed,filename):
    gauinv_good = do_gauinv(rand_good_out)
    do_plot_cdf(rand_good_out, nbins, str(current_seed) + '_randGDv2.png') 
    do_plot_cdf(gauinv_good, nbins, str(current_seed) + '_randGDv2.png')
    do_plot_compareSTA(target,avgFun_good,nlag,filename)  

#%%version 3
def addLines_v3(filepath,nx,ny,nz,xsiz,ysiz,zsiz):
    #del first i lines
    line1 = "SISIM SIMULATIONS:    \n"
    line2 = "3 "+str(nx)+" "+str(ny)+" "+str(nz)+" "+str(xsiz/2)+" "+str(ysiz/2)+" "+str(zsiz/2)+" "+str(xsiz)+" "+str(ysiz)+" "+str(zsiz)+" 1\n"
    #3:3 type data and its value. 1: relation number
    line3 = "Simulated Value\n"
    line4 = "random number\n"
    line5 = "path_index\n"
    with open(filepath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        f.write(line1)
        f.write(line2)
        f.write(line3)
        f.write(line4)
        f.write(line5)
        
        f.writelines(lines)
        f.close()

#def change_geo_obj_par(nx,ny,nz,filepath):
#def change_rand2loc_par(nx,ny,nz,xsiz,ysiz,zsiz,filepath):    
def getRand2LocData(filepath):
    with open(filepath,'r') as f:
        lines = f.readlines()        
        f.close()
    res = splitData(lines[5])
    return res[1]
        

def f_v3(t, target,inv1,inv2, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,current_seed,order,  fac,connect,nlag):   
    gd = inv1*math.cos(t) + inv2*math.sin(t)    
    rand = do_inv2gcum(gd)
    sis_out_i,rand_out_i = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,current_seed,rand,order)
    #filename = 'conGd'#the file name without file type(.out)
    #xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = do_connect(fac,sis_out_i,filename,connect,nx,ny,nz,xsiz,ysiz,zsiz, nlag)
    addLines_v3(exe_path + 'sis_inv.out',nx,ny,nz,xsiz,ysiz,zsiz)    
    os.system('echo geo_obj.par | .\geo_obj.exe')#use sisim.out to create geo_obj.dat
    os.system('echo rank2loc.par | .\\rank2loc.exe ')#use geo_obj.dat to create 
    res = getRand2LocData(exe_path + 'rank2loc.out')    
    #return res,xyzAvgFun,sis_out_i,rand_out_i
    
    return abs(target-res),sis_out_i,rand_out_i
def do_plot_fv3(x_opt,y_opt,filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x_opt, y_opt, 'b^--', alpha=0.6, linewidth=1,label='|| target - f(x) ||')
    plt.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('|| target - f(x) ||')
    
    plt.xticks(np.linspace(0,x_opt,9))
    
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=dpi)
    plt.show()
    
#%% version 5
# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    #generate an initial point
    best = bounds[:,0] + rand(len(bounds))*(bounds[:,1]-bounds[:,0])
    #evaluate the initial point
    best_eval = objective(best)
    #current working solution
    curr, curr_eval = best, best_eval
    #generate an empty vector to store track of scores
    scores = []
    for i in range(n_iterations):
        #take a step
        candidate = curr + randn(len(bounds))*step_size
        #evaluate candidate point
        candidate_eval = objective(candidate)
        #check for new best solution
        if candidate_eval < best_eval:
            #store new best point
            best, best_eval = candidate, candidate_eval
            #keep track of scores
            scores.append(best_eval)
        #difference between candidate and current point evalution
        diff = candidate_eval - curr_eval
        #calculate tempture for current epoch
        t = temp / float(i + 1)
        #calculate metropplis acceptance criterion
        metropolis = exp(-diff/t)
        #check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            #store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best,best_eval,scores]
def changeSisInvSeed(seed):    
    old_seed = r"-random number seed"
    new_seed = str(seed) + '  ' + old_seed + '\n'    
    file_path = exe_path + "sisim_inv.par"
    with open(file_path,"r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i in range(len(lines)):            
            if old_seed in lines[i]:
                lines[i] = new_seed        
            f.write(lines[i])
        f.close()     
#%% rank2loc        
def objective(t,target,inv1,inv2, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    gd = inv1*math.cos(t) + inv2*math.sin(t)    
    rand = do_inv2gcum(gd)#back-transformed into a uniform vector
    sis_out_i,rand_out_i = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order)
    addLines_v3(exe_path + 'sis_inv.out',nx,ny,nz,xsiz,ysiz,zsiz)   
    changeGeoPar('sis_inv.out')
    os.system('echo geo_obj.par | .\geo_obj.exe')#use sisim.out to create geo_obj.dat
    os.system('echo rank2loc.par | .\\rank2loc.exe ')#use geo_obj.dat to create 
    res = getRand2LocData(exe_path + 'rank2loc.out')    
    return abs(target-res),sis_out_i,rand_out_i


#%% version 6
def target_v6(nz):
    res_vec = []
    for i in range(nz):
        filename = 'rank2loc_' + str(i+1) + '.par'
        os.system('echo '+filename+' | .\\rank2loc.exe ')#use geo_obj.dat to create 
        res = getRand2LocData(exe_path + 'rank2loc_'+str(i+1)+'.out') 
        res_vec.append(res)
    return res_vec

def objective_v6(t,target,inv1,inv2, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):  
    gd = inv1*math.cos(t) + inv2*math.sin(t)     
    rand = do_inv2gcum(gd) 
    sis_out_i,rand_out_i = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order) 
    addLines_v3(exe_path + 'sis_inv.out',nx,ny,nz,xsiz,ysiz,zsiz)    
    changeGeoPar('sis_inv.out')
    os.system('echo geo_obj.par | .\geo_obj.exe')#use sisim.out to create geo_obj.dat
    res_vec = target_v6(nz)
    dif_fun, dif_res = Optimization.getDiffBetween(target,res_vec)               
    return dif_res,res_vec,sis_out_i,rand_out_i
#%% for getting a target
def changeGeoPar(filename):
    filepath = exe_path + 'geo_obj.par'
    with open(filepath,'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        f.writelines(lines[:4])
        f.write(filename+'\n')
        f.writelines(lines[5:])
        f.close()
        
def setTargetConnect3d(fac,nlag,connect, threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed):
    '''
    Set a random path and the target function
    :param fac: phase of interest 
    :param nlag: max distance of lag
    :param connect: connectivity analysis type
    :param threads: phase type
    :param gcdf: global distribution
    :param nx: x number of model
    :param xsiz: cell size
    :param ny: y number of model
    :param ysiz: cell size
    :param nz: z number of model
    :param zsiz: cell size
    :param seed: random seed
    
    Return: random path, target function
    '''
    
    sis_out,rand_out,order = do_sisim_gd(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed)
    #save reference model as file
    file_path = exe_path + str(seed) +'_referencemodel.gslib'
    with open(file_path, "w") as f:
       f.seek(0)
       f.truncate()
       f.write("SISIM SIMULATIONS:\n")
       f.write("3 100 60 20 0.5 0.5 0.5 1 1 1 1\n")
       f.write("Simulated Value\n")
       f.write("Random Number\n")
       f.write("Path Index\n")

       n = len(sis_out)
       for i in range(0, n):
           f.write(str(sis_out[i])+' '+str(rand_out[i])+' '+str(order[i])+"\n")
       f.close()
        
    filename = 'conGd'#the file name without file type(.out)
    xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = do_connect(fac,sis_out,filename,connect,nx,ny,nz,xsiz,ysiz,zsiz,nlag)
    return order,xyzAvgFun

def setTargetRank2loc(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed):
    sis_out,rand_out,order = do_sisim_gd(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed)
    addLines_v3(exe_path + str(seed) +'gd.out',nx,ny,nz,xsiz,ysiz,zsiz)   
    changeGeoPar(str(seed) +'gd.out')
    os.system('echo geo_obj.par | .\geo_obj.exe')#use sisim.out to create geo_obj.dat
    os.system('echo rank2loc.par | .\\rank2loc.exe ')#use geo_obj.dat to create 
    target = getRand2LocData(exe_path + 'rank2loc.out')   
    return order,target

def setTargetVerticalRank2loc_2(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed, target_seed_vec):
    idx = -1
    while True:
        if idx == -1:
            sis_out,rand_out,order = do_sisim_gd(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed)
            addLines_v3(exe_path + str(seed) +'gd.out',nx,ny,nz,xsiz,ysiz,zsiz)        
            changeGeoPar(str(seed) +'gd.out')
        else:
            sis_out,rand_out,order = do_sisim_gd(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, target_seed_vec[idx])
            addLines_v3(exe_path + str(target_seed_vec[idx]) +'gd.out',nx,ny,nz,xsiz,ysiz,zsiz)        
            changeGeoPar(str(target_seed_vec[idx]) +'gd.out')
        os.system('echo geo_obj.par | .\geo_obj.exe')    
        os.system('echo rank2loc.par | .\\rank2loc.exe ')#use geo_obj.dat to create 
        target = getRand2LocData(exe_path + 'rank2loc.out')   
        idx = idx + 1    
        if target == 0:
            continue
        if idx == len(target_seed_vec):
            return 'cant find a target bigger than zero!'
        target_vertical = []
        ex = 0
        for i in range(nz):
            filename = 'rank2loc_' + str(i+1) + '.par'
            f_rank2loc = open(exe_path + 'rank2loc.par', 'r')               
            lines = f_rank2loc.readlines() 
            f_rank2loc.close()
            
            f = open(exe_path + filename, 'w')
            f.seek(0)
            f.truncate()
            f.writelines(lines[:5])    
            f.write('rank2loc_' + str(i + 1) +'.out               -Output Ranking file\n')
            f.writelines(lines[6:11])     
            info1 = '1                         -  number of blocks for first well i\n'
            info2 = '              -     x,y,z location\n'
            well1 = '26   51    '
            well2 = '30   45    '
            f.write(info1)
            f.write(well1 + str(i+1) + info2)
            f.write(info1)
            f.write(well2 + str(i+1) + info2)
            f.close()
            os.system('echo '+filename+' | .\\rank2loc.exe ')#use geo_obj.dat to create 
            target = getRand2LocData(exe_path + 'rank2loc_'+str(i+1)+'.out') 
            target_vertical.append(target)            
            if target == 0:
                ex += 1           
        if len(target_vertical) == nz and ex <= 5:           
            return order,target_vertical
    return 0,[]
        
def setTargetVerticalRank2loc(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed, target_seed_vec):
    order,target = setTargetRank2loc(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed)
    i = 0
    while target == 0:
        order,target = setTargetRank2loc(threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, target_seed_vec[i])
        i += 1
        if i == len(target_seed_vec):
            return 'cant find a target bigger than zero!'
    target_vertical = []
    for i in range(nz):
        filename = 'rank2loc_' + str(i+1) + '.par'
        f_rank2loc = open(exe_path + 'rank2loc.par', 'r')               
        lines = f_rank2loc.readlines() 
        f_rank2loc.close()
        
        f = open(exe_path + filename, 'w')
        f.seek(0)
        f.truncate()
        f.writelines(lines[:5])    
        f.write('rank2loc_' + str(i + 1) +'.out               -Output Ranking file\n')
        f.writelines(lines[6:11])     
        info1 = '1                         -  number of blocks for first well i\n'
        info2 = '              -     x,y,z location\n'
        well1 = '26   51    '
        well2 = '30   45    '
        f.write(info1)
        f.write(well1 + str(i+1) + info2)
        f.write(info1)
        f.write(well2 + str(i+1) + info2)
        f.close()
        os.system('echo '+filename+' | .\\rank2loc.exe ')#use geo_obj.dat to create 
        target = getRand2LocData(exe_path + 'rank2loc_'+str(i+1)+'.out') 
        target_vertical.append(target)
    return order,target_vertical



def saveRes(sis,rand,order,filepath):
    '''
    Save SIS result to filepath
    :param sis: vector of SIS result 
    :param rand: vector of SIS random numbers 
    :param order: vector of SIS random path
    :param filepath: absolute path of file 
    
    '''
    with open(filepath, "w") as f:
        f.seek(0)
        f.truncate()
        f.write("SISIM SIMULATIONS:\n")
        f.write("3 100 60 20 0.5 0.5 0.5 1 1 1 1\n")
        f.write("Simulated Value\n")
        f.write("Random Number\n")
        f.write("Path Index\n")

        n = len(sis)
        for i in range(0, n):
            f.write(str(sis[i])+' '+str(rand[i])+' '+str(order[i])+"\n")
        f.close()
#%% golden search
def goldenSearch(a, b, e,  inv1, inv2,fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    #[a,b]: bounds
    #e: stop criteria
    iter_time = 0
    a1 = b-0.618*(b-a)
    a2 = a+0.618*(b-a)
    #f1,f2 = f(a1),f(a2)
    f1,sis1,rand1,fun1 = golden_f(inv1, inv2, a1, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
    f2,sis2,rand2,fun2 = golden_f(inv1, inv2, a2, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
    
    while abs(b-a)>e:
        if f1<f2:
            b,a2,f2= a2,a1,f1
            a1 = b-0.618*(b-a)
            #f1 = f(a1)            
            f1,sis1,rand1,fun1 = golden_f(inv1, inv2, a1, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
        else:
            a,a1,f1=a1,a2,f2
            a2 = a+0.618*(b-a)
            #f2 = f(a2)        
            f2,sis2,rand2,fun2 = golden_f(inv1, inv2, a2, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
        iter_time = iter_time + 1
    a = (a+b)/2
    print("黄金切割法下的极值点为a* = {:.4f}".format(a))
    print('本次迭代次数：'+str(iter_time))
    return a

def golden_f(inv1, inv2, t, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    #t: deformation parameter
    inv = inv1*math.cos(t) + inv2*math.sin(t)
    rand = do_inv2gcum(inv)
    sis,rand = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order)      
    filename = 'conGd'#the file name without file type(.out)
    xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = do_connect(fac,sis,filename,connect,nx,ny,nz,xsiz,ysiz,zsiz, nlag)
    dif_fun, dif_res = Optimization.getDiffBetween(target, xyzAvgFun)
    return dif_res,sis,rand,xyzAvgFun




