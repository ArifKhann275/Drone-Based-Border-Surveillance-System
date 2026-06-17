import math, random, json
import tkinter as tk
from tkinter import messagebox

ROWS, COLS = 10, 10
STATIONS = [(0,0),(0,9),(9,0),(9,9)]
ZONE_SIZE = 2

ALPHA=0.6; GAMMA=0.3; EPSILON=0.5
ETA=0.1; LAMBDA=7.0; MU=1.2; K_TURN=0.2
VISIT_PENALTY=10.0
ENERGY_PER_CELL=4
SAFETY_BUFFER=1.3

def make_grid(seed=42):
    random.seed(seed)
    g={}
    for r in range(ROWS):
        for c in range(COLS):
            edge=min(r,c,ROWS-1-r,COLS-1-c)
            pri=3 if edge==0 else(2 if edge==1 else 1)
            g[(r,c)]={"covered":0,"priority":pri,
                      "risk":round(random.uniform(0.1,0.9),2),
                      "threat":False,"is_station":False,
                      "visits":0,"threat_detected":False}
    for s in STATIONS: g[s]["is_station"]=True;g[s]["priority"]=0
    return g

def is_st(r,c): return (r,c) in STATIONS
def nearest_st(r,c): return min(STATIONS,key=lambda s:abs(r-s[0])+abs(c-s[1]))
def get_zone_id(r,c): return (r//ZONE_SIZE,c//ZONE_SIZE)
def get_zone_cells(zr,zc):
    out=[]
    for r in range(zr*ZONE_SIZE,min((zr+1)*ZONE_SIZE,ROWS)):
        for c in range(zc*ZONE_SIZE,min((zc+1)*ZONE_SIZE,COLS)):
            if not is_st(r,c): out.append((r,c))
    return out
def get_all_zones():
    return [(zr,zc) for zr in range(math.ceil(ROWS/ZONE_SIZE))
                     for zc in range(math.ceil(COLS/ZONE_SIZE))]

def nbrs(r,c,g):
    out=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<ROWS and 0<=nc<COLS and not g[(nr,nc)]["is_station"]:
            out.append((nr,nc))
    return out

def D_eff(r1,c1,r2,c2,pr,pc):
    D=math.sqrt((r2-r1)**2+(c2-c1)**2)
    turn=0 if pr is None else(0 if(r2-r1==r1-pr and c2-c1==c1-pc)else 1)
    return round(D*(1+K_TURN*turn),3),turn

def W_eff(r1,c1,r2,c2,ws=12,wd=90):
    h=math.degrees(math.atan2(c2-c1,r2-r1))%360
    diff=abs(wd-h)%360
    if diff>180: diff=360-diff
    return round(ws*(diff/180)*(1+50/200),3)

def E_ret(r,c):
    dist=min(abs(r-s[0])+abs(c-s[1]) for s in STATIONS)
    return round(dist*ENERGY_PER_CELL*SAFETY_BUFFER,2)

def should_recharge(r,c,battery):
    st=nearest_st(r,c)
    dist=abs(r-st[0])+abs(c-st[1])
    e_return=dist*ENERGY_PER_CELL*SAFETY_BUFFER
    return battery<=(e_return+ENERGY_PER_CELL), round(e_return,2)

def calc_C(covered_pct):
    if covered_pct >= 100: return 0.0
    elif covered_pct == 0: return 2.0   
    else: return 1.0   

def zone_score(zr,zc,dr,dc,g,detected,step):
    W1=3.0;W2=4.0;W3=0.5;W4=0.2
    cells=get_zone_cells(zr,zc)
    if not cells: return -999
    avg_cov=sum(g[c]["covered"] for c in cells)/len(cells)
    incompletion=(100-avg_cov)/100
    has_threat=any(g[c]["threat"] and c not in detected for c in cells)
    threat_b=2.0 if has_threat else 0
    last_v=min((g[c]["visits"] for c in cells),default=0)
    tgap=max(0,step-last_v*10)/100
    zrc=zr*ZONE_SIZE+ZONE_SIZE//2; zcc=zc*ZONE_SIZE+ZONE_SIZE//2
    travel=abs(dr-zrc)+abs(dc-zcc)
    return round(W1*incompletion+W2*threat_b+W3*tgap-W4*travel,4)

def rank_zones(dr,dc,g,detected,step):
    scored=[]
    for (zr,zc) in get_all_zones():
        cells=get_zone_cells(zr,zc)
        avg=sum(g[c]["covered"] for c in cells)/len(cells) if cells else 0
        sc=zone_score(zr,zc,dr,dc,g,detected,step)
        scored.append({"zone":(zr,zc),"score":sc,"cov":round(avg,1)})
    scored.sort(key=lambda x:x["score"],reverse=True)
    return scored

def cost_fn(r1,c1,r2,c2,pr,pc,g,detected,target_zone=None):
    d,turn=D_eff(r1,c1,r2,c2,pr,pc)
    w=W_eff(r1,c1,r2,c2)
    R=g[(r2,c2)]["risk"]
    er=E_ret(r2,c2)
    C=calc_C(g[(r2,c2)]["covered"])
    P=g[(r2,c2)]["priority"]
    done_penalty=20.0 if g[(r2,c2)]["covered"]>=100 else 0
    V=VISIT_PENALTY*g[(r2,c2)]["visits"]+done_penalty
    tc=12.0 if(g[(r2,c2)]["threat"] and (r2,c2) in detected) else (-4.0 if g[(r2,c2)]["threat"] else 0)
    zb=-2.0 if(target_zone and get_zone_id(r2,c2)==target_zone) else 0
    cost=ALPHA*d+GAMMA*w+EPSILON*R+ETA*er+V+tc+zb-LAMBDA*C-MU*P
    return round(cost,4),{
        "D_eff":d,"W_eff":w,"R":R,"E_return":er,
        "C":C,"P":P,"V":V,"turn":turn,"cost":round(cost,4)}

def smart_move(r,c,pr,pc,g,detected,target_zone=None):
    ns=nbrs(r,c,g)
    if not ns: return r,c,{}
    best,bpos,bbd=float('inf'),(r,c),{}
    for nr,nc in ns:
        cv,bd=cost_fn(r,c,nr,nc,pr,pc,g,detected,target_zone)
        if cv<best: best=cv;bpos=(nr,nc);bbd=bd
    g[bpos]["visits"]+=1
    return bpos[0],bpos[1],bbd

def move_toward(r,c,tr,tc_):
    nr,nc=r,c
    if r!=tr: nr=r+(1 if tr>r else -1)
    elif c!=tc_: nc=c+(1 if tc_>c else -1)
    if is_st(nr,nc): return r,c
    return nr,nc

def gps_snake(r,c,d):
    nc=c+d;nr=r
    if nc<0 or nc>=COLS or is_st(r,nc):
        nr=(r+1)%ROWS;d=-d;nc=c+d
    if 0<=nr<ROWS and 0<=nc<COLS and is_st(nr,nc): nc+=d
    if not(0<=nc<COLS): nc=c
    return nr,nc,d

def find_uncovered_in_zone(zr,zc,g):
    cells=get_zone_cells(zr,zc)
    empties=[c for c in cells if g[c]["covered"]==0]
    partials=[c for c in cells if 0<g[c]["covered"]<100]
    if empties: return empties[0]
    if partials: return partials[0]
    return None

class DroneSimGUI:
    def __init__(self,root):
        self.root=root
        self.root.title("Drone v5 — Empty-First Coverage")
        self.root.configure(bg="#1a1a2e")
        self.cell=44
        self.reset_state()
        self.create_widgets()
        self.draw_grids()

    def reset_state(self):
        self.gs=make_grid(42);self.gg=make_grid(42)
        self.threats=[(1,8),(5,3),(8,6)]
        for t in self.threats:
            self.gs[t]["threat"]=True;self.gg[t]["threat"]=True
        self.sr,self.sc=1,1;self.gr,self.gc=1,1
        self.sb=self.gb=100;self.pr,self.pc=None,None
        self.dr=1;self.sR=self.gR=0;self.step=0;self.log=[]
        self.running=False;self.detected=set();self.speed=300
        self.target_zone=None;self.target_cell=None
        self.post_recharge=False;self.zone_rankings=[]

    def create_widgets(self):
        top=tk.Frame(self.root,bg="#16213e",pady=6);top.pack(fill="x")
        self.step_lbl=tk.Label(top,text="Step:0/150",font=("Consolas",12,"bold"),fg="#e2e2e2",bg="#16213e")
        self.step_lbl.pack(side="left",padx=10)
        self.diff_lbl=tk.Label(top,text="Cov:—",font=("Consolas",11),fg="#f0c040",bg="#16213e")
        self.diff_lbl.pack(side="left",padx=10)
        self.threat_lbl=tk.Label(top,text="Threats:0/3",font=("Consolas",11),fg="#ff6b6b",bg="#16213e")
        self.threat_lbl.pack(side="left",padx=10)
        self.zone_lbl=tk.Label(top,text="Zone:—",font=("Consolas",11),fg="#aaffff",bg="#16213e")
        self.zone_lbl.pack(side="left",padx=10)
        mf=tk.Frame(self.root,bg="#1a1a2e");mf.pack(padx=10,pady=4)
        sf=tk.Frame(mf,bg="#0f3460",bd=2,relief="groove");sf.pack(side="left",padx=6)
        tk.Label(sf,text="🧠 SMART",font=("Arial",10,"bold"),fg="#00d2ff",bg="#0f3460").pack(pady=3)
        self.sc_=tk.Canvas(sf,width=COLS*self.cell,height=ROWS*self.cell,bg="#0a0a1a")
        self.sc_.pack()
        self.s_stat=tk.Label(sf,text="Bat:100% | RC:0 | Cov:0%",font=("Consolas",9),fg="#aaffaa",bg="#0f3460")
        self.s_stat.pack(pady=2)
        gf=tk.Frame(mf,bg="#3d0000",bd=2,relief="groove");gf.pack(side="left",padx=6)
        tk.Label(gf,text="🛰️ GPS",font=("Arial",10,"bold"),fg="#ff9999",bg="#3d0000").pack(pady=3)
        self.gc_=tk.Canvas(gf,width=COLS*self.cell,height=ROWS*self.cell,bg="#0a0a1a")
        self.gc_.pack()
        self.g_stat=tk.Label(gf,text="Bat:100% | RC:0 | Cov:0%",font=("Consolas",9),fg="#ffaaaa",bg="#3d0000")
        self.g_stat.pack(pady=2)
        zf=tk.Frame(self.root,bg="#1a1a2e");zf.pack(fill="x",padx=10,pady=1)
        tk.Label(zf,text="Rankings:",font=("Consolas",9),fg="#888",bg="#1a1a2e").pack(side="left")
        self.rank_lbl=tk.Label(zf,text="—",font=("Consolas",9),fg="#ffdd57",bg="#1a1a2e",wraplength=750,justify="left")
        self.rank_lbl.pack(side="left",padx=5)
        self.info_lbl=tk.Label(self.root,text="—",font=("Consolas",9),fg="#ffaa44",bg="#1a1a2e")
        self.info_lbl.pack()
        self.cost_lbl=tk.Label(self.root,text="Cost:—",font=("Consolas",9),fg="#888",bg="#1a1a2e")
        self.cost_lbl.pack()
        cf=tk.Frame(self.root,bg="#1a1a2e",pady=4);cf.pack()
        self.btn_run=tk.Button(cf,text="▶ Start",font=("Arial",11,"bold"),bg="#27ae60",fg="white",width=9,command=self.toggle_run)
        self.btn_run.pack(side="left",padx=5)
        tk.Button(cf,text="⏭ Step",font=("Arial",10),bg="#2980b9",fg="white",width=7,command=self.do_step).pack(side="left",padx=5)
        tk.Button(cf,text="↺ Reset",font=("Arial",10),bg="#7f8c8d",fg="white",width=7,command=self.reset).pack(side="left",padx=5)
        self.speed_var=tk.IntVar(value=300)
        tk.Scale(cf,from_=50,to=1000,orient="horizontal",variable=self.speed_var,bg="#1a1a2e",fg="white",troughcolor="#333",length=110).pack(side="left")

    def draw_grids(self):
        self.sc_.delete("all");self.gc_.delete("all")
        cs=self.cell
        target_cells=set(get_zone_cells(*self.target_zone)) if self.target_zone else set()
        for r in range(ROWS):
            for c in range(COLS):
                x1,y1=c*cs,r*cs;x2,y2=x1+cs,y1+cs
                mx,my=x1+cs//2,y1+cs//2
                z=self.gs[(r,c)]
                if z["is_station"]: col="#f39c12"
                elif (r,c) in target_cells: col="#004488" if z["covered"]<100 else "#0a4a2a"
                elif z["threat"] and z["threat_detected"]: col="#8e44ad"
                elif z["threat"]: col="#c0392b"
                elif z["covered"]>=100: col="#1abc9c"
                elif z["covered"]>0: col="#148f77"
                else: col="#0d1b2a"
                self.sc_.create_rectangle(x1,y1,x2,y2,fill=col,outline="#2a2a3e")
                gz=self.gg[(r,c)]
                gcol="#f39c12" if gz["is_station"] else ("#c0392b" if gz["threat"] else ("#2980b9" if gz["covered"]>=100 else ("#1a5276" if gz["covered"]>0 else "#0d1b2a")))
                self.gc_.create_rectangle(x1,y1,x2,y2,fill=gcol,outline="#2a2a3e")
        sx,sy=self.sc*cs+cs//2,self.sr*cs+cs//2
        self.sc_.create_oval(sx-13,sy-13,sx+13,sy+13,fill="#f0c040");self.sc_.create_text(sx,sy,text="✈")
        gx,gy=self.gc*cs+cs//2,self.gr*cs+cs//2
        self.gc_.create_oval(gx-13,gy-13,gx+13,gy+13,fill="#e74c3c");self.gc_.create_text(gx,gy,text="✈")

    def select_zone(self):
        rankings=rank_zones(self.sr,self.sc,self.gs,self.detected,self.step)
        self.zone_rankings=rankings
        best=rankings[0]
        self.target_zone=best["zone"]
        self.target_cell=find_uncovered_in_zone(*best["zone"],self.gs)
        self.zone_lbl.config(text=f"Zone:{best['zone']}")
        return best

    def do_step(self):
        if self.step>=150: self.finish();return
        self.step+=1
        must_rc,_=should_recharge(self.sr,self.sc,self.sb)
        if must_rc:
            st=nearest_st(self.sr,self.sc);self.sr,self.sc=st;self.sb=100;self.sR+=1;self.post_recharge=True
        elif self.post_recharge:
            best=self.select_zone();self.post_recharge=False
        else:
            if self.target_zone:
                if all(self.gs[c]["covered"]>=100 for c in get_zone_cells(*self.target_zone)): best=self.select_zone()
            if self.target_cell and get_zone_id(self.sr,self.sc)!=self.target_zone:
                tr,tc_=self.target_cell;self.sr,self.sc=move_toward(self.sr,self.sc,tr,tc_);bd={}
            else:
                self.sr,self.sc,bd=smart_move(self.sr,self.sc,self.pr,self.pc,self.gs,self.detected,self.target_zone)
            cell=self.gs[(self.sr,self.sc)];cell["covered"]=min(100,cell["covered"]+45);self.sb=max(0,self.sb-ENERGY_PER_CELL)
            if cell["threat"] and (self.sr,self.sc) not in self.detected: self.detected.add((self.sr,self.sc));cell["threat_detected"]=True
        self.pr,self.pc=self.sr,self.sc
        if self.gb<=20: self.gr,self.gc=nearest_st(self.gr,self.gc);self.gb=100;self.gR+=1
        else: self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr);self.gg[(self.gr,self.gc)]["covered"]=min(100,self.gg[(self.gr,self.gc)]["covered"]+45);self.gb=max(0,self.gb-ENERGY_PER_CELL)
        self.draw_grids()

    def toggle_run(self):
        self.running=not self.running
        if self.running: self.auto_run()

    def auto_run(self):
        if self.running and self.step<150:
            self.do_step();self.root.after(self.speed_var.get(),self.auto_run)

    def reset(self):
        self.reset_state();self.draw_grids()

    def finish(self):
        messagebox.showinfo("Done", "Simulation Finished")

if __name__=="__main__":
    root=tk.Tk()
    app=DroneSimGUI(root)
    root.mainloop()
