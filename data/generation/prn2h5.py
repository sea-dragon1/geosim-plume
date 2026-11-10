#!/usr/bin/env python
'''
    将生成的所有数据prn转化为h5格式，并打包在一起
'''
import os
import sys
import struct
from pathlib import Path
import numpy as np
import h5py


def prn2h5_one(prn_path, save_path, prnlst):


    pr_ls  = []
    vx_ls  = []
    vy_ls  = []
    exx_ls = []
    eyy_ls = []
    exy_ls = []
    sxx_ls = []
    syy_ls = []
    sxy_ls = []
    ro_ls  = []
    nu_ls  = []
    tk_ls  = []
    t_ls = []


    keys = ['pr', 'vx', 'vy', 'ro', 'nu', 'tk'] # , 'exx', 'eyy', 'exy', 'sxx', 'syy', 'sxy'
    vals = [pr_ls, vx_ls, vy_ls, ro_ls, nu_ls, tk_ls] #, exx_ls, eyy_ls, exy_ls, sxx_ls, syy_ls, sxy_ls
    ky_dict = {key: val for key, val in zip(keys, vals)}

    for j in range(len(prnlst)):
        print("Load Data from %s ... "%prnlst[j]);
        with open(prnlst[j],"rb") as f:
            header1=f.read(4)
            sizint, sizlng, sizflt, sizdbl = struct.unpack('<bbbb',header1)
            print("sizint = ", sizint, " sizlng = ", sizlng, " sizflt = ", sizflt, " sizdbl = ", sizdbl)
        
            header2=f.read(5*sizlng)
            if sizlng == 4:
                xnumx, ynumy, mnumx, mnumy, marknum = struct.unpack('<lllll',header2)
            elif sizlng == 8:
                xnumx, ynumy, mnumx, mnumy, marknum = struct.unpack('<qqqqq',header2)
            else:
                print('Unknown data format for LONG INTEGER.')
                
            print("xnumx = ", xnumx, " ynumy = ", ynumy, " mnumx = ", mnumx, " mnumy = ", mnumy, " marknum = ", marknum)

            header3=f.read(3*sizdbl)
            if sizdbl == 8:
                xsize, ysize, pinit = struct.unpack('<ddd',header3)
            else:
                print('Unknown data format for DOUBLE.')
                
            print("xsize = ", xsize, " ysize = ", ysize, " pinit = ", pinit)
            print()
            x_coordinate = np.linspace(0,xsize,xnumx)
            y_coordinate = np.linspace(0,ysize,ynumy)

            #header3=f.read(4*sizdbl)
            f.seek(4*sizdbl,1)
        
            header3=f.read(2*sizdbl)
            if sizdbl == 8:
                GXKOEF, GYKOEF = struct.unpack('<dd',header3)
            else:
                print('Unknown data format for DOUBLE.')
                
            print("GXKOEF = ", GXKOEF, " GYKOEF = ", GYKOEF)
        
            header4=f.read(sizint*2+sizlng+sizdbl)
            rocknum, bondnum, n0, ival1 = struct.unpack('<iqid',header4)
            print("rocknum = ", rocknum, " bondnum = ", bondnum, " n0 = ", n0, " ival1 = ", ival1)
            t_ls.append(ival1)
            # /* Nodes Num */
            nodenum=xnumx*ynumy
        
            ## /* Cells Num */
            #cellnum=(xnumx-1)*(ynumy-1);
        
            ## /* Koef for processing */
            # xstpx=xsize/(xnumx-1.);
            # ystpy=ysize/(ynumy-1.);


            #kfx=1.0/xstpx;
            #kfy=1.0/ystpy;
            #kfxx=kfx*kfx;
            #kfyy=kfy*kfy;
            #kfxy=kfx*kfy;
        
            ## /* Marker size */
            #mardx=xstpx/mnumx;
            #mardy=ystpy/mnumy;
        
            ## /* Step for numerical differentiation */
            #numdx=5e-1*mardx;
            #numdy=5e-1*mardy;
        
            ## /* Spec counters */
            #nodenum2=nodenum*2;
            #nodenum3=nodenum*3;
            #xnumx1=xnumx-1;
            #ynumy1=ynumy-1;
        
            pr  = np.zeros(nodenum, dtype=np.float32)
            vx  = np.zeros(nodenum, dtype=np.float32)
            vy  = np.zeros(nodenum, dtype=np.float32)
            exx = np.zeros(nodenum, dtype=np.float32)
            eyy = np.zeros(nodenum, dtype=np.float32)
            exy = np.zeros(nodenum, dtype=np.float32)
            sxx = np.zeros(nodenum, dtype=np.float32)
            syy = np.zeros(nodenum, dtype=np.float32)
            sxy = np.zeros(nodenum, dtype=np.float32)
            ro  = np.zeros(nodenum, dtype=np.float32)
            nu  = np.zeros(nodenum, dtype=np.float32)
            tk  = np.zeros(nodenum, dtype=np.float32)
            # /* Skip Rock Types information */
            f.seek( sizint*rocknum + sizdbl*rocknum*24 ,1)
        
            if sizlng == 4:
                for i in range(nodenum):
                    data=f.read(4*sizlng+22*sizflt)
                    pr[i],vx[i],vy[i],t,t,t, exx[i],eyy[i],exy[i],sxx[i],syy[i],sxy[i],ro[i],nu[i], t,t,t,t,t,t,t, tk[i],t,t,t,t = struct.unpack('<ffflllfffffffffffffffflfff',data)
            elif sizlng == 8:
                for i in range(nodenum):
                    data=f.read(4*sizlng+22*sizflt)
                    pr[i],vx[i],vy[i],t,t,t, exx[i],eyy[i],exy[i],sxx[i],syy[i],sxy[i],ro[i],nu[i], t,t,t,t,t,t,t, tk[i],t,t,t,t = struct.unpack('<fffqqqffffffffffffffffqfff',data)
            else:
                print('Unknown data format for LONG INTEGER.')

            #ro.tofile("ro.dat")



            # ## tk
            tk.shape=(xnumx,ynumy)
            tk_ls.append(tk)
            print('shape of tk:', np.shape(tk))
            # plt.figure(figsize=(16,8))
            # plt.imshow(tk.T, cmap=plt.get_cmap('jet'), aspect='equal')
            # #plt.show()
            # plt.savefig("tk_%s.png"%(prnlst[j]))
    
            # ## nu
            nu.shape=(xnumx,ynumy)
            ltnu=np.log10(nu)
            nu_ls.append(ltnu)
            # plt.figure(figsize=(16,8))
            # plt.imshow(ltnu.T, cmap=plt.get_cmap('jet'), aspect='equal')
            # #plt.show()
            # plt.savefig("nu_%s.png"%(prnlst[j]))
            
            # ## pr
            pr.shape=(xnumx,ynumy)
            pr_ls.append(pr)
            # plt.figure(figsize=(16,8))
            # plt.imshow(pr.T, cmap=plt.get_cmap('jet'), aspect='equal')
            # #plt.show()
            # plt.savefig("pr_%s.png"%(prnlst[j]))
        
            # ## ro
            ro.shape=(xnumx,ynumy)
            ro_ls.append(ro)
            # rmin = ro.min()
            # rmax = ro.max()
            # rmean = ro.mean()
            # rrad=rmax-rmean

            # rclipmax=rmax
            # rclipmin=rmean-2*rrad

            # plt.figure(figsize=(16,8))
            # plt.imshow(ro.T, cmap=plt.get_cmap('jet'), vmin=rclipmin, vmax=rclipmax, aspect='equal')
            # #plt.show()
            # plt.savefig("ro_%s.png"%(prnlst[j]))
            
    #        #tk sub
    #        tks = tk.T[0:200,750:1250]
    #        plt.figure(figsize=(16,8))
    #        plt.imshow(tks, cmap=plt.get_cmap('jet'), aspect='equal', vmin=0, vmax=2000)
    #        plt.savefig("tks_%s.png"%(prnlst[j]))

    #        #vx,vy
            vx.shape=(xnumx,ynumy)
            vy.shape=(xnumx,ynumy)
            vx_ls.append(vx)
            vy_ls.append(vy)
    #        
    #        vxs1 = vx.T[0:200,750:1250]
    #        vys1 = vy.T[0:200,750:1250]
    #
    #        fig, (ax1,ax2) = plt.subplots(2,figsize=(16,16))
    #        ax1.imshow(vxs1, cmap=plt.get_cmap('jet'), aspect='equal')
    #        ax2.imshow(vys1, cmap=plt.get_cmap('jet'), aspect='equal')
    #        #plt.show()
    #        plt.savefig("vxs_vys_%s.png"%(prnlst[j]))

    #        #quiver sub
    #        X = np.arange(500)
    #        Y = np.arange(200)
    #        U,V = np.meshgrid(X,Y)
    #        
    #        us  =    U[::10,::50]
    #        vs  =    V[::10,::50]
    #        vxs = vx.T[0:200:10,750:1250:50]
    #        vys = vy.T[0:200:10,750:1250:50]
    #
    #        vxy=np.sqrt(vx**2+vy**2)
    #        vxymax=vxy.max()
    #
    #        fig, ax = plt.subplots(figsize=(16,8),dpi=160)
    #        im = ax.imshow(tks, cmap=plt.get_cmap('jet'), vmin=0, vmax=2000)
    #        #q = ax.quiver(us,vs,vxs,vys,color='white',angles='uv',scale_units='inches',scale=1e-7,width=0.0015)
    #        q = ax.quiver(us,vs,vxs,vys,color='white',angles='xy',units='xy',scale_units='inches',scale=vxymax*0.5,width=1.25,pivot='mid')
    #        
    #        # Set x,y ticks
    #        plt.xticks(np.linspace(0,500,6,endpoint=True),np.linspace(750,1250,6,endpoint=True, dtype=np.int32))
    #        plt.yticks(np.linspace(0,200,9,endpoint=True))
    #        
    #        # Set x,y labels
    #        plt.xlabel("X(km)")
    #        plt.ylabel("Y(km)")
    #        
    #        plt.title("Temperature field")
    #        plt.savefig("tkv_%s.png"%(prnlst[j]))

    #        #quiver all
    #        XA = np.arange(xnumx)
    #        YA = np.arange(ynumy)
    #        UA,VA = np.meshgrid(XA,YA)
    #        
    #        uas  =   UA[::25,::75]
    #        vas  =   VA[::25,::75]
    #        vxas = vx.T[::25,::75]
    #        vyas = vy.T[::25,::75]
    #        
    #        fig, ax = plt.subplots(figsize=(16,8))
    #        q = ax.quiver(uas,vas,vxas,vyas,angles='xy',color='white',units='xy',scale_units='inches',scale=vxymax*1.75,width=2.5,pivot='mid')
    #        im = ax.imshow(tk.T, cmap=plt.get_cmap('jet'), vmin=0, vmax=2000)
    #        
    #        # Set x,y ticks
    #        plt.xticks(np.linspace(0,2000,9,endpoint=True),np.linspace(0,4000,9,endpoint=True, dtype=np.int32))
    #        plt.yticks(np.linspace(0,400,5,endpoint=True),np.linspace(0,800,5,endpoint=True, dtype=np.int32))
    #        
    #        # Set x,y labels
    #        plt.xlabel("X(km)")
    #        plt.ylabel("Y(km)")
    #        
    #        plt.title("Temperature field")
    #        #plt.show()
    #        plt.savefig("tkv_all_%s.png"%(prnlst[j]))
    t_ls = np.array(t_ls)
    x_coordinate = np.array(x_coordinate)
    y_coordinate = np.array(y_coordinate)
    return t_ls, x_coordinate, y_coordinate, ky_dict

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True,
                    help="campaign directory containing the case folders")
    ap.add_argument("--prefix", default="run",
                    help="case prefix (default: run; use T or T_test for Plume1)")
    ap.add_argument("--indices", default=None,
                    help="case indices, e.g. 0-127 (default: discover matching folders)")
    ap.add_argument("--last-decade", type=int, default=1270,
                    help="last plumeN.prn decade (930 for the 97-frame Plume1 train pool)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: <workdir>/datah5)")
    ap.add_argument("--combined-name", default=None,
                    help="default: <prefix>_combined.h5")
    a = ap.parse_args()

    workdir = Path(a.workdir)
    outdir = Path(a.outdir) if a.outdir else workdir / "datah5"
    outdir.mkdir(parents=True, exist_ok=True)
    if a.indices:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from common import parse_indices
        indices = parse_indices(a.indices)
    else:
        indices = []
        for p in workdir.iterdir():
            if not p.is_dir() or not p.name.startswith(a.prefix):
                continue
            tail = p.name[len(a.prefix):]
            if tail.isdigit():
                indices.append(int(tail))
        indices.sort()
    if not indices:
        raise SystemExit("no case folders found under {}".format(workdir))

    all_ky_data = {}
    all_t_coordinates = []
    x_coordinate = None
    y_coordinate = None
    for i in indices:
        case = workdir / "{}{}".format(a.prefix, i)
        prn_path = str(case)
        prnlst = [os.path.join(prn_path, name) for name in
                  ("plume0.prn", "plume1.prn", "plume2.prn", "plume3.prn")]
        prnlst += [os.path.join(prn_path, "plume{}.prn".format(k))
                   for k in range(10, a.last_decade + 1, 10)]
        missing = [p for p in prnlst if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("missing PRN files for case {}: {}".format(i, missing[:3]))
        t_ls, x_tmp, y_tmp, ky_dict = prn2h5_one(
            prn_path, str(outdir / "{}_{}.h5".format(a.prefix, i)), prnlst)
        if x_coordinate is None:
            x_coordinate = x_tmp
        if y_coordinate is None:
            y_coordinate = y_tmp
        all_t_coordinates.append(t_ls)
        for key, data in ky_dict.items():
            all_ky_data.setdefault(key, []).append(np.array(data))

    for key in all_ky_data:
        all_ky_data[key] = np.stack(all_ky_data[key], axis=0)
    all_t_coordinates = np.array(all_t_coordinates)
    combined = outdir / (a.combined_name or "{}_combined.h5".format(a.prefix))
    with h5py.File(combined, "w") as h5file:
        for key, data in all_ky_data.items():
            h5file.create_dataset(key, data=data)
        h5file.create_dataset("t-coordinate", data=all_t_coordinates)
        h5file.create_dataset("x-coordinate", data=x_coordinate)
        h5file.create_dataset("y-coordinate", data=y_coordinate)
    print("wrote", combined, "cases", len(indices), "frames", len(all_t_coordinates[0]))


if __name__ == "__main__":
    main()
