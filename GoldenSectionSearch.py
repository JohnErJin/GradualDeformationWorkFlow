# -*- coding: utf-8 -*-
'''
Import required package
'''
from matplotlib import pyplot
import numpy as np                                        # arrays and matrix math
from gd_sis_utils import *
import random
#%%
'''
Sequential indicator simulation parameter
'''
nx,xsiz,ny,ysiz,nz,zsiz = 100, 1, 60, 1, 20, 1
fac,  connect = 1, 6
threads,gcdf = [0,1,2], [0.3, 0.5, 0.2]
nlag = 20
n_iterations = 1000
objfun_vec = []
sis_good_vec = []
rand_good_vec = []
AvgFun_vec = []
#%%
'''
Seed number
'''
seed = 56734#reference model seed
if seed <= n_iterations:#set random seed
    seed_vec = random.sample(range(seed,n_iterations),n_iterations+2)
else:
    seed_vec = random.sample(range(n_iterations,seed),n_iterations+2)
#%%
'''
Set fixed random path and target cure
'''
order,target = setTargetConnect3d(fac,nlag,connect, threads, gcdf, nx, xsiz, ny, ysiz, nz, zsiz, seed)
#%%
'''
Extract two vectors of random numbers from Sequential Indicator Simulation
'''
seed_vec[0] = 15113#initial model seed
sis_out1,rand_out1 = do_sis_gdf(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed_vec[0],order)#S1  
sis_out2,rand_out2 = do_sis_gdf(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed_vec[1],order)#S2
#%%
'''
Transform two vectors of random numbers into Gaussian vectors
'''
inv1 = do_gauinv(rand_out1)  #using the inverse cumulative Gaussian distribution
inv2 = do_gauinv(rand_out2)  
#%%
'''
Choose a initla model
''' 
filename = 'conGd'#the file name without file type(.out)
xloc1,xFun1,yloc1,yFun1,zloc1,zFun1,xyzAvgLoc1,fun1 = do_connect(fac,sis_out1,filename,connect,nx,ny,nz, xsiz,ysiz,zsiz, nlag)
xloc2,xFun2,yloc2,yFun2,zloc2,zFun2,xyzAvgLoc2,fun2 = do_connect(fac,sis_out2,filename,connect,nx,ny,nz, xsiz,ysiz,zsiz, nlag)
dif_fun1, dif_res1 = Optimization.getDiffBetween(target, fun1)        
dif_fun2, dif_res2 = Optimization.getDiffBetween(target, fun2)        

do_plot_compareSTA(target,fun1,nlag,'initial.png', 'initial')
objfun_vec.append(dif_res1)
sis_good_vec.append(sis_out1)
rand_good_vec.append(rand_out1)
initial_obj = dif_res1
initial_inv = inv1
inv2 = inv2

#%%golden section search algorithm 
def gs(obj, a, b, e,  inv1, inv2,fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    #[a,b]: bounds
    #e: stop criteria
    iter_time = 1
    a1 = b-0.618*(b-a)
    a2 = a+0.618*(b-a)
    #f1,f2 = f(a1),f(a2)
    idx_vec = []
    obj_vec = []
    sis_vec = []
    rand_vec = []
    fun_vec = []
    f1,sis1,rand1,fun1 = gf(inv1, inv2, a1, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
    idx_vec.append(a1)
    obj_vec.append(f1)
    sis_vec.append(sis1)
    rand_vec.append(rand1)    
    fun_vec.append(fun1)
    f2,sis2,rand2,fun2 = gf(inv1, inv2, a2, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
    idx_vec.append(a2)
    obj_vec.append(f2)
    sis_vec.append(sis2)
    rand_vec.append(rand2)    
    fun_vec.append(fun2)
    
    while abs(b-a)>e:
        if f2 < obj or f1 < obj:
           min_index = obj_vec.index(min(obj_vec))
           r_opt = idx_vec[min_index]
           sis = sis_vec[min_index]
           rand = rand_vec[min_index]
           fun = fun_vec[min_index]
           return r_opt,idx_vec,obj_vec,sis,rand,fun
       
        if f1<f2:
            b,a2,f2= a2,a1,f1
            a1 = b-0.618*(b-a)       
            f1,sis1,rand1,fun1 = gf(inv1, inv2, a1, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
            idx_vec.append(a1)
            obj_vec.append(f1)
            sis_vec.append(sis1)
            rand_vec.append(rand1)
            fun_vec.append(fun1)
        else:
            a,a1,f1=a1,a2,f2
            a2 = a+0.618*(b-a)
            #f2 = f(a2)        
            f2,sis2,rand2,fun2 = gf(inv1, inv2, a2, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)
            idx_vec.append(a2)
            obj_vec.append(f2)
            sis_vec.append(sis2)
            rand_vec.append(rand2)
            fun_vec.append(fun2)
            
        iter_time = iter_time + 1
    min_index = obj_vec.index(min(obj_vec))
    r_opt = idx_vec[min_index]
    sis = sis_vec[min_index]
    rand = rand_vec[min_index]
    fun = fun_vec[min_index]
    return r_opt,idx_vec,obj_vec,sis,rand,fun

def gf(inv1, inv2, t, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order):
    #t: deformation parameter
    inv = inv1*math.cos(t) + inv2*math.sin(t)
    rand = do_inv2gcum(inv)
    sis,rand = do_sis_inv(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,rand,order)      
    filename = 'conGd'#the file name without file type(.out)
    xloc,xFun,yloc,yFun,zloc,zFun,xyzAvgLoc,xyzAvgFun = do_connect(fac,sis,filename,connect,nx,ny,nz,xsiz,ysiz,zsiz, nlag)
    dif_fun, dif_res = Optimization.getDiffBetween(target, xyzAvgFun)
    return dif_res,sis,rand,xyzAvgFun

def showGoldenSectionSearchProcess(local_idx, local_obj,filename):
    minindex = local_obj.index(min(local_obj))
    r_opt = local_idx[minindex]    
    plt.figure(figsize=(8, 6))
    plt.scatter(local_idx, local_obj, color='blue', label='r')
    plt.scatter(0, obj,color='red', label='r_initial')
    plt.scatter(r_opt, min(local_obj), color='green', label='r_minimum')
    plt.legend()
    plt.xlabel('Deformation parameter')
    plt.ylabel('Objective function')       
    plt.xlim((LEFT, RIGHT))    
    plt.yscale('log')
    figure_path = exe_path + 'img/' + filename
    plt.savefig(figure_path, dpi=1000,bbox_inches = 'tight')
    plt.show()
#%% 
obj = initial_obj
times = 0
idx=0
local_idx_vec,local_obj_vec=[],[]
LEFT,RIGHT=-math.pi,math.pi
while obj > 1e-2:   
    r_opt,local_idx,local_obj,sis_good,rand_good,xyzAvgFun = gs(obj, LEFT, RIGHT, 1e-8, initial_inv, inv2, fac,connect,nlag,target, threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed,order)   
    if min(local_obj) < obj:
        showGoldenSectionSearchProcess(local_idx, local_obj, 'GoldenSectionSearchProcess_Iterations_'+ str(idx + 1) +'.png')
        do_plot_compareSTA(target_fun=target,fun=xyzAvgFun,nlag=20,filename='iter'+ str(idx + 1) +'.png',labelname='iteration '+str(idx+1))
        saveRes(sis_good,rand_good,order,exe_path + 'SimulatedResult/Iter' + str(idx + 1) + '.gslib')
       
        local_idx_vec.append(local_idx)
        local_obj_vec.append(local_obj)
        objfun_vec.append(min(local_obj))
        sis_good_vec.append(sis_good)
        rand_good_vec.append(rand_good)
        AvgFun_vec.append(xyzAvgFun)
        
        obj = min(local_obj)
        if idx > 5:
            plt.figure(figsize=(8, 6))
            x = range(0, len(objfun_vec))
            y = objfun_vec
            plt.plot(x, y, 'bo-', linewidth=1)
            plt.xlabel('Iterations')
            plt.ylabel('Objective functions')
            plt.grid(True)
            plt.yscale('log')
            plt.xticks(range(0, len(objfun_vec)))
            figure_path = exe_path + 'img/' + 'iterations_objLog_times' + str(idx) + '.png' 
            plt.savefig(figure_path, dpi=dpi,bbox_inches = 'tight')
            
        
        sis_out1,rand_out1 = sis_good,rand_good
        sis_out2,rand_out2 = do_sis_gdf(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed_vec[times+2],order)#S2
        initial_inv = do_gauinv(rand_out1)
        inv2 = do_gauinv(rand_out2)
        idx = idx + 1
        
    if min(local_obj) > obj:
        print("cannot find a better result.")
        print("do again , start with another u2!")
        sis_out2,rand_out2 = do_sis_gdf(threads,gcdf,nx,xsiz,ny,ysiz,nz,zsiz,seed_vec[times+2],order)#S2
        initial_inv = do_gauinv(rand_out1)  #using the inverse cumulative Gaussian distribution
        inv2 = do_gauinv(rand_out2)
    #limit the iteration times
    times = times + 1
    if times == 10:
        continue
