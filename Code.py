import math, random, json
import tkinter as tk
from tkinter import messagebox

ROWS, COLS = 10, 10
STATIONS = [(0,0),(0,9),(9,0),(9,9)]
ZONE_SIZE = 2

# SMRS Weights
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

# ── Physics & Utility ─────────────────────────────────────────
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

# ── Baseline 2: GREEDY ALGORITHM ──────────────────────────────
def greedy_move(r, c, pr, pc, g, detected):
    
    ns = nbrs(r, c, g)
    if not ns: return r, c
    
    best_score = -float('inf')
    best_pos = (r, c)
    
    for nr, nc in ns:
        cell = g[(nr, nc)]
        
        cov_reward = (100 - cell["covered"]) 
        threat_reward = 50 if cell["threat"] and (nr, nc) not in detected else 0
        pri_reward = cell["priority"] * 10
        penalty = cell["visits"] * 30
        
        score = cov_reward + threat_reward + pri_reward - penalty
        
        if score > best_score:
            best_score = score
            best_pos = (nr, nc)
            
    g[best_pos]["visits"] += 1
    return best_pos[0], best_pos[1]

# ── Proposed: SMRS (Semantic Mission Resume System) ───────────
def zone_score(zr,zc,dr,dc,g,detected,step):
    W1=3.0; W2=4.0; W3=0.5; W4=0.2
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

    tc=12.0 if(g[(r2,c2)]["threat"] and (r2,c2) in detected) else \
       (-4.0 if g[(r2,c2)]["threat"] else 0)

    zb=-2.0 if(target_zone and get_zone_id(r2,c2)==target_zone) else 0

    cost=ALPHA*d+GAMMA*w+EPSILON*R+ETA*er+V+tc+zb-LAMBDA*C-MU*P
    return round(cost,4),{"cost":round(cost,4)}

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
        self.root.title("Drone Surveillance: SMRS vs Greedy vs GPS (3-Way Comparison)")
        self.root.configure(bg="#1a1a2e")
        self.cell=32 
        self.reset_state()
        self.create_widgets()
        self.draw_grids()

    def reset_state(self):
        # 3 Grids for 3 algorithms
        self.gs=make_grid(42)  # SMRS
        self.ggre=make_grid(42) # Greedy
        self.ggps=make_grid(42) # GPS

        self.threats=[(1,8),(5,3),(8,6)]
        for t in self.threats:
            self.gs[t]["threat"]=True; 
            self.ggre[t]["threat"]=True
            self.ggps[t]["threat"]=True
            
        self.sr,self.sc=1,1; self.grer,self.grec=1,1; self.gr,self.gc=1,1
        self.sb=100; self.greb=100; self.gb=100
        self.s_pr,self.s_pc=None,None; self.gre_pr,self.gre_pc=None,None
        self.dr=1; 
        self.sR=0; self.greR=0; self.gR=0
        
        self.step=0; self.log=[]
        self.running=False
        
        self.s_detected=set()
        self.gre_detected=set()
        
        self.speed=300
        self.target_zone=None
        self.target_cell=None
        self.post_recharge=False

    def create_widgets(self):
        top=tk.Frame(self.root,bg="#16213e",pady=6);top.pack(fill="x")
        self.step_lbl=tk.Label(top,text="Step:0/300",
            font=("Consolas",12,"bold"),fg="#e2e2e2",bg="#16213e")
        self.step_lbl.pack(side="left",padx=10)
        
        mf=tk.Frame(self.root,bg="#1a1a2e");mf.pack(padx=10,pady=4)

        # 1. SMRS Frame
        sf=tk.Frame(mf,bg="#0f3460",bd=2,relief="groove");sf.pack(side="left",padx=4)
        tk.Label(sf,text="🧠 3. Proposed SMRS",font=("Arial",10,"bold"),fg="#00d2ff",bg="#0f3460").pack(pady=3)
        self.sc_=tk.Canvas(sf,width=COLS*self.cell,height=ROWS*self.cell,bg="#0a0a1a");self.sc_.pack()
        self.s_stat=tk.Label(sf,text="Bat:100% | RC:0 | Cov:0%",font=("Consolas",9),fg="#aaffaa",bg="#0f3460")
        self.s_stat.pack(pady=2)

        # 2. Greedy Frame
        gref=tk.Frame(mf,bg="#4a235a",bd=2,relief="groove");gref.pack(side="left",padx=4)
        tk.Label(gref,text="🎯 2. Greedy Baseline",font=("Arial",10,"bold"),fg="#d7bde2",bg="#4a235a").pack(pady=3)
        self.grec_=tk.Canvas(gref,width=COLS*self.cell,height=ROWS*self.cell,bg="#0a0a1a");self.grec_.pack()
        self.gre_stat=tk.Label(gref,text="Bat:100% | RC:0 | Cov:0%",font=("Consolas",9),fg="#ebdef0",bg="#4a235a")
        self.gre_stat.pack(pady=2)

        # 3. GPS Frame
        gf=tk.Frame(mf,bg="#3d0000",bd=2,relief="groove");gf.pack(side="left",padx=4)
        tk.Label(gf,text="🛰️ 1. GPS Snake",font=("Arial",10,"bold"),fg="#ff9999",bg="#3d0000").pack(pady=3)
        self.gc_=tk.Canvas(gf,width=COLS*self.cell,height=ROWS*self.cell,bg="#0a0a1a");self.gc_.pack()
        self.g_stat=tk.Label(gf,text="Bat:100% | RC:0 | Cov:0%",font=("Consolas",9),fg="#ffaaaa",bg="#3d0000")
        self.g_stat.pack(pady=2)

        cf=tk.Frame(self.root,bg="#1a1a2e",pady=4);cf.pack()
        self.btn_run=tk.Button(cf,text="▶ Start",font=("Arial",10,"bold"),bg="#27ae60",fg="white",width=9,command=self.toggle_run)
        self.btn_run.pack(side="left",padx=5)
        tk.Button(cf,text="⏭ Step",font=("Arial",10),bg="#2980b9",fg="white",width=7,command=self.do_step).pack(side="left",padx=5)
        tk.Button(cf,text="↺ Reset",font=("Arial",10),bg="#7f8c8d",fg="white",width=7,command=self.reset).pack(side="left",padx=5)

    def draw_grid_on_canvas(self, canvas, grid, r_drone, c_drone, drone_col, drone_icon, draw_zones=False):
        canvas.delete("all")
        cs=self.cell
        target_cells=set(get_zone_cells(*self.target_zone)) if (draw_zones and self.target_zone) else set()

        for r in range(ROWS):
            for c in range(COLS):
                x1,y1=c*cs,r*cs;x2,y2=x1+cs,y1+cs
                mx,my=x1+cs//2,y1+cs//2
                z=grid[(r,c)]

                col="#0d1b2a"
                if z["is_station"]: col="#f39c12"
                elif draw_zones and (r,c) in target_cells:
                    if z["covered"]>=100: col="#0a4a2a"
                    elif z["covered"]==0: col="#003366"  
                    else: col="#004488"  
                elif z["threat"] and z.get("threat_detected", False): col="#8e44ad"
                elif z["threat"]: col="#c0392b"
                elif z["covered"]>=100: col="#1abc9c" if draw_zones else ("#8e44ad" if drone_icon=="G" else "#2980b9")
                elif z["covered"]>0: col="#148f77" if draw_zones else ("#5b2c6f" if drone_icon=="G" else "#1a5276")

                border="#00aaff" if(draw_zones and (r,c) in target_cells and not z["is_station"]) else "#2a2a3e"
                bw=2 if border!="#2a2a3e" else 1
                canvas.create_rectangle(x1,y1,x2,y2,fill=col,outline=border,width=bw)

                if z["is_station"]: canvas.create_text(mx,my,text="⚡",font=("Arial",10))
                elif z["threat"] and z.get("threat_detected", False): canvas.create_text(mx,my,text="✅",font=("Arial",9))
                elif z["threat"]: canvas.create_text(mx,my,text="🚨",font=("Arial",9))

        if draw_zones:
            for zr in range(math.ceil(ROWS/ZONE_SIZE)+1):
                canvas.create_line(0,zr*ZONE_SIZE*cs,COLS*cs,zr*ZONE_SIZE*cs,fill="#334455",dash=(2,4))
            for zc in range(math.ceil(COLS/ZONE_SIZE)+1):
                canvas.create_line(zc*ZONE_SIZE*cs,0,zc*ZONE_SIZE*cs,ROWS*cs,fill="#334455",dash=(2,4))

        dx,dy=c_drone*cs+cs//2,r_drone*cs+cs//2
        canvas.create_oval(dx-10,dy-10,dx+10,dy+10,fill=drone_col,outline="#fff",width=1)
        canvas.create_text(dx,dy,text="✈",font=("Arial",10,"bold"),fill="#fff" if drone_col!="#f0c040" else "#000")

    def draw_grids(self):
        self.draw_grid_on_canvas(self.sc_, self.gs, self.sr, self.sc, "#f0c040", "S", draw_zones=True)
        self.draw_grid_on_canvas(self.grec_, self.ggre, self.grer, self.grec, "#d7bde2", "G")
        self.draw_grid_on_canvas(self.gc_, self.ggps, self.gr, self.gc, "#e74c3c", "P")

    def do_step(self):
        if self.step>=300: self.finish();return
        self.step+=1

        # ── 1. SMRS LOGIC ──
        must_rc, e_ret = should_recharge(self.sr, self.sc, self.sb)
        if must_rc:
            st = nearest_st(self.sr, self.sc)
            self.sr, self.sc = st; self.sb = 100; self.sR += 1
            self.post_recharge = True
        elif self.post_recharge:
            best = rank_zones(self.sr,self.sc,self.gs,self.s_detected,self.step)[0]
            self.target_zone = best["zone"]
            self.target_cell = find_uncovered_in_zone(*best["zone"],self.gs)
            self.post_recharge = False
        else:
            if self.target_zone:
                zr, zc = self.target_zone
                zone_cells = get_zone_cells(zr, zc)
                if all(self.gs[c]["covered"] >= 100 for c in zone_cells):
                    best = rank_zones(self.sr,self.sc,self.gs,self.s_detected,self.step)[0]
                    self.target_zone = best["zone"]

            if self.target_cell and get_zone_id(self.sr,self.sc)!=self.target_zone:
                tr, tc_ = self.target_cell
                nr, nc = move_toward(self.sr, self.sc, tr, tc_)
                self.sr, self.sc = nr, nc
            else:
                self.sr, self.sc, bd = smart_move(self.sr, self.sc, self.s_pr, self.s_pc, self.gs, self.s_detected, self.target_zone)

            cell = self.gs[(self.sr, self.sc)]
            cell["covered"] = min(100, cell["covered"] + 100)
            self.sb = max(0, self.sb - ENERGY_PER_CELL)
            if cell["threat"] and (self.sr, self.sc) not in self.s_detected:
                self.s_detected.add((self.sr, self.sc))
                cell["threat_detected"] = True

            if self.target_zone:
                self.target_cell = find_uncovered_in_zone(*self.target_zone, self.gs)

        self.s_pr, self.s_pc = self.sr, self.sc

        # ── 2. GREEDY LOGIC ──
        must_rc_gre, _ = should_recharge(self.grer, self.grec, self.greb)
        if must_rc_gre:
            st = nearest_st(self.grer, self.grec)
            self.grer, self.grec = st; self.greb = 100; self.greR += 1
        else:
            self.grer, self.grec = greedy_move(self.grer, self.grec, self.gre_pr, self.gre_pc, self.ggre, self.gre_detected)
            cell = self.ggre[(self.grer, self.grec)]
            cell["covered"] = min(100, cell["covered"] + 100)
            self.greb = max(0, self.greb - ENERGY_PER_CELL)
            if cell["threat"] and (self.grer, self.grec) not in self.gre_detected:
                self.gre_detected.add((self.grer, self.grec))
                cell["threat_detected"] = True
        self.gre_pr, self.gre_pc = self.grer, self.grec

        # ── 3. GPS LOGIC ──
        if self.gb<=20:
            gt=nearest_st(self.gr,self.gc)
            self.gr,self.gc=gt;self.gb=100;self.gR+=1
        else:
            self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr)
            self.ggps[(self.gr,self.gc)]["covered"]=min(100,self.ggps[(self.gr,self.gc)]["covered"]+100)
            self.gb=max(0,self.gb-ENERGY_PER_CELL)

        # Stats Update
        sc_=round(sum(1 for v in self.gs.values() if v["covered"]>=100)/(ROWS*COLS)*100,1)
        grec_=round(sum(1 for v in self.ggre.values() if v["covered"]>=100)/(ROWS*COLS)*100,1)
        gc_=round(sum(1 for v in self.ggps.values() if v["covered"]>=100)/(ROWS*COLS)*100,1)
        
        st_=len(self.s_detected)
        gret_=len(self.gre_detected)
        gt_=sum(1 for t in self.threats if self.ggps[t]["covered"]>=50)

        self.log.append({"step":self.step, "s_cov":sc_, "gre_cov":grec_, "g_cov":gc_,
                         "s_bat":self.sb, "gre_bat":self.greb, "g_bat":self.gb,
                         "s_thr":st_, "gre_thr":gret_, "g_thr":gt_})

        self.step_lbl.config(text=f"Step:{self.step}/300")
        self.s_stat.config(text=f"Bat:{self.sb}% | RC:{self.sR} | Cov:{sc_}% | Thr:{st_}/3")
        self.gre_stat.config(text=f"Bat:{self.greb}% | RC:{self.greR} | Cov:{grec_}% | Thr:{gret_}/3")
        self.g_stat.config(text=f"Bat:{self.gb}% | RC:{self.gR} | Cov:{gc_}% | Thr:{gt_}/3")
        
        self.draw_grids()

    def toggle_run(self):
        self.running=not self.running
        self.btn_run.config(text="⏸ Pause" if self.running else "▶ Start",
            bg="#e67e22" if self.running else "#27ae60")
        if self.running: self.auto_run()

    def auto_run(self):
        if self.running and self.step<300:
            self.do_step()
            self.root.after(self.speed, self.auto_run)
        elif self.step>=300:
            self.running=False
            self.btn_run.config(text="▶ Start",bg="#27ae60")
            self.finish()

    def reset(self):
        self.running=False
        self.btn_run.config(text="▶ Start",bg="#27ae60")
        self.reset_state()
        self.draw_grids()

    def finish(self):
        self.running=False
        last=self.log[-1] if self.log else {}
        try:
            with open('drone_3way_comparison.json','w') as f:
                json.dump(self.log,f,indent=2,default=str)
        except Exception: pass
        
        msg=(f"📊 FINAL RESULTS (300 Steps):\n\n"
             f"🧠 SMRS (Proposed)  → Cov: {last.get('s_cov',0)}%  | Thr: {last.get('s_thr',0)}/3 | RC: {self.sR}\n"
             f"🎯 Greedy (Baseline)→ Cov: {last.get('gre_cov',0)}%  | Thr: {last.get('gre_thr',0)}/3 | RC: {self.greR}\n"
             f"🛰️ GPS (Baseline)   → Cov: {last.get('g_cov',0)}%  | Thr: {last.get('g_thr',0)}/3 | RC: {self.gR}\n\n"
             f"💾 Saved to drone_3way_comparison.json")
        messagebox.showinfo("Simulation Complete",msg)

if __name__=="__main__":
    root=tk.Tk()
    app=DroneSimGUI(root)
    root.mainloop()
