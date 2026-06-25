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
                      "visits":0,"threat_detected":False,
                      "last_visited_step":0}
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

# ── Physics ───────────────────────────────────────────────────
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

def calc_C(covered_pct):
    if covered_pct>=100: return 0.0
    elif covered_pct==0: return 2.0
    else: return 1.0

# ════════════════════════════════════════════════════════════
# TWO-HOP ENERGY FEASIBILITY CHECK
# Formula: E_needed = E(current→target) + E(target→station) + buffer
# ════════════════════════════════════════════════════════════
def energy_to_travel(r1,c1,r2,c2,ws=12,wd=90):
    dist=abs(r1-r2)+abs(c1-c2)
    if dist==0: return 0.0
    heading=math.degrees(math.atan2(c2-c1,r2-r1))%360
    diff=abs(wd-heading)%360
    if diff>180: diff=360-diff
    wind_factor=1.0+(diff/180)*0.3
    return round(dist*ENERGY_PER_CELL*wind_factor,2)

def two_hop_check(drone_r,drone_c,target_r,target_c,battery):
    """
    ══════════════════════════════════════════════════════
    TWO-HOP ENERGY FEASIBILITY
    E_needed = E(current→target) + E(target→station) + buffer
    battery >= E_needed → যাও ✅
    battery <  E_needed → station যাও ❌
    ══════════════════════════════════════════════════════
    """
    e_hop1=energy_to_travel(drone_r,drone_c,target_r,target_c)
    st=nearest_st(target_r,target_c)
    e_hop2=energy_to_travel(target_r,target_c,st[0],st[1])
    e_needed=e_hop1+e_hop2+ENERGY_PER_CELL
    return battery>=e_needed, round(e_hop1,1), round(e_hop2,1), round(e_needed,1)

def must_recharge_now(r,c,battery):
    """
    পরের step এ আর যেতে পারবো না কিনা check করো।
    battery <= E(current→station) + 1_step_buffer
    """
    st=nearest_st(r,c)
    e_ret=energy_to_travel(r,c,st[0],st[1])
    return battery<=(e_ret+ENERGY_PER_CELL), round(e_ret,1), st

# ── SMRS functions ────────────────────────────────────────────
def zone_score(zr,zc,dr,dc,g,detected,step):
    W1=3.0;W2=4.0;W3=0.8;W4=0.2
    cells=get_zone_cells(zr,zc)
    if not cells: return -999
    avg_cov=sum(g[c]["covered"] for c in cells)/len(cells)
    incompletion=(100-avg_cov)/100
    has_threat=any(g[c]["threat"] and c not in detected for c in cells)
    threat_b=2.0 if has_threat else 0
    # Real time gap: last_visited_step দিয়ে
    last_step=min((g[c]["last_visited_step"] for c in cells),default=0)
    time_gap=min((step-last_step)/100,1.0)
    zrc=zr*ZONE_SIZE+ZONE_SIZE//2
    zcc=zc*ZONE_SIZE+ZONE_SIZE//2
    travel=abs(dr-zrc)+abs(dc-zcc)
    return round(W1*incompletion+W2*threat_b+W3*time_gap-W4*travel,4)

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
    done_pen=20.0 if g[(r2,c2)]["covered"]>=100 else 0
    V=VISIT_PENALTY*g[(r2,c2)]["visits"]+done_pen
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

# ── KEY FIX: move_toward ─────────────────────────────────────
def move_toward(r,c,tr,tc_):
    """
    Target এর দিকে এক ধাপ যাও।
    FIX: target নিজেই station হলে ঢুকতে দাও,
         কিন্তু পথে অন্য station এ ঢুকবে না।
    """
    nr,nc=r,c
    if r!=tr: nr=r+(1 if tr>r else -1)
    elif c!=tc_: nc=c+(1 if tc_>c else -1)
    # Target নিজেই station হলে ঢুকতে দাও
    # কিন্তু পথে অন্য station এ ঢুকবে না
    if is_st(nr,nc) and (nr,nc)!=(tr,tc_):
        return r,c
    return nr,nc

def find_uncovered_in_zone(zr,zc,g):
    cells=get_zone_cells(zr,zc)
    empties=[c for c in cells if g[c]["covered"]==0]
    if empties: return empties[0]
    partials=[c for c in cells if 0<g[c]["covered"]<100]
    return partials[0] if partials else None

# ── Greedy baseline ───────────────────────────────────────────
def greedy_move(r,c,pr,pc,g,detected):
    ns=nbrs(r,c,g)
    if not ns: return r,c
    best_score=-float('inf');best_pos=(r,c)
    for nr,nc in ns:
        cell=g[(nr,nc)]
        cov_reward=(100-cell["covered"])
        threat_reward=50 if cell["threat"] and (nr,nc) not in detected else 0
        pri_reward=cell["priority"]*10
        penalty=cell["visits"]*30
        score=cov_reward+threat_reward+pri_reward-penalty
        if score>best_score: best_score=score;best_pos=(nr,nc)
    g[best_pos]["visits"]+=1
    return best_pos[0],best_pos[1]

# ── GPS Snake ─────────────────────────────────────────────────
def gps_snake(r,c,d):
    nc=c+d;nr=r
    if nc<0 or nc>=COLS or is_st(r,nc):
        nr=(r+1)%ROWS;d=-d;nc=c+d
    if 0<=nr<ROWS and 0<=nc<COLS and is_st(nr,nc): nc+=d
    if not(0<=nc<COLS): nc=c
    return nr,nc,d

# ════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════
class DroneSimGUI:
    def __init__(self,root):
        self.root=root
        self.root.title("Drone Surveillance: SMRS vs Greedy vs GPS")
        self.root.configure(bg="#1a1a2e")
        self.cell=32
        self.reset_state()
        self.create_widgets()
        self.draw_grids()

    def reset_state(self):
        self.gs=make_grid(42)
        self.ggre=make_grid(42)
        self.ggps=make_grid(42)
        self.threats=[(1,8),(5,3),(8,6)]
        for t in self.threats:
            self.gs[t]["threat"]=True
            self.ggre[t]["threat"]=True
            self.ggps[t]["threat"]=True
        self.sr,self.sc=1,1; self.grer,self.grec=1,1; self.gr,self.gc=1,1
        self.sb=100; self.greb=100; self.gb=100
        self.s_pr,self.s_pc=None,None
        self.gre_pr,self.gre_pc=None,None
        self.dr=1
        self.sR=0; self.greR=0; self.gR=0
        self.step=0; self.log=[]; self.running=False
        self.s_detected=set(); self.gre_detected=set()
        self.speed=200
        self.target_zone=None; self.target_cell=None
        self.post_recharge=False
        # Station navigation state
        self.going_to_station=False
        self.station_target=None

    def create_widgets(self):
        top=tk.Frame(self.root,bg="#16213e",pady=6);top.pack(fill="x")
        self.step_lbl=tk.Label(top,text="Step:0/300",
            font=("Consolas",12,"bold"),fg="#e2e2e2",bg="#16213e")
        self.step_lbl.pack(side="left",padx=10)
        self.info_lbl=tk.Label(top,text="—",
            font=("Consolas",10),fg="#ffdd57",bg="#16213e")
        self.info_lbl.pack(side="left",padx=10)

        mf=tk.Frame(self.root,bg="#1a1a2e");mf.pack(padx=10,pady=4)

        # SMRS
        sf=tk.Frame(mf,bg="#0f3460",bd=2,relief="groove");sf.pack(side="left",padx=4)
        tk.Label(sf,text="🧠 SMRS (Proposed)",
            font=("Arial",10,"bold"),fg="#00d2ff",bg="#0f3460").pack(pady=3)
        self.sc_=tk.Canvas(sf,width=COLS*self.cell,
            height=ROWS*self.cell,bg="#0a0a1a");self.sc_.pack()
        self.s_stat=tk.Label(sf,text="Bat:100% | RC:0 | Cov:0% | Thr:0/3",
            font=("Consolas",8),fg="#aaffaa",bg="#0f3460")
        self.s_stat.pack(pady=2)

        # Greedy
        gref=tk.Frame(mf,bg="#4a235a",bd=2,relief="groove");gref.pack(side="left",padx=4)
        tk.Label(gref,text="🎯 Greedy (Baseline)",
            font=("Arial",10,"bold"),fg="#d7bde2",bg="#4a235a").pack(pady=3)
        self.grec_=tk.Canvas(gref,width=COLS*self.cell,
            height=ROWS*self.cell,bg="#0a0a1a");self.grec_.pack()
        self.gre_stat=tk.Label(gref,text="Bat:100% | RC:0 | Cov:0% | Thr:0/3",
            font=("Consolas",8),fg="#ebdef0",bg="#4a235a")
        self.gre_stat.pack(pady=2)

        # GPS
        gf=tk.Frame(mf,bg="#3d0000",bd=2,relief="groove");gf.pack(side="left",padx=4)
        tk.Label(gf,text="🛰️ GPS Snake (Baseline)",
            font=("Arial",10,"bold"),fg="#ff9999",bg="#3d0000").pack(pady=3)
        self.gc_=tk.Canvas(gf,width=COLS*self.cell,
            height=ROWS*self.cell,bg="#0a0a1a");self.gc_.pack()
        self.g_stat=tk.Label(gf,text="Bat:100% | RC:0 | Cov:0% | Thr:0/3",
            font=("Consolas",8),fg="#ffaaaa",bg="#3d0000")
        self.g_stat.pack(pady=2)

        # Two-hop info bar
        self.hop_lbl=tk.Label(self.root,
            text="Two-Hop: E_hop1=— E_hop2=— E_needed=— Battery=—",
            font=("Consolas",8),fg="#888888",bg="#1a1a2e")
        self.hop_lbl.pack(pady=1)

        leg=tk.Frame(self.root,bg="#1a1a2e");leg.pack(pady=2)
        for sym,lbl in [("🟩","Covered"),("⬛","Empty"),("🟥","Threat"),
                        ("🟣","Detected"),("🟨","Station"),("🔵","Target zone")]:
            tk.Label(leg,text=f"{sym}{lbl}",font=("Arial",8),
                fg="#ccc",bg="#1a1a2e").pack(side="left",padx=4)

        cf=tk.Frame(self.root,bg="#1a1a2e",pady=4);cf.pack()
        self.btn_run=tk.Button(cf,text="▶ Start",font=("Arial",10,"bold"),
            bg="#27ae60",fg="white",width=9,command=self.toggle_run)
        self.btn_run.pack(side="left",padx=5)
        tk.Button(cf,text="⏭ Step",font=("Arial",10),bg="#2980b9",
            fg="white",width=7,command=self.do_step).pack(side="left",padx=5)
        tk.Button(cf,text="↺ Reset",font=("Arial",10),bg="#7f8c8d",
            fg="white",width=7,command=self.reset).pack(side="left",padx=5)
        tk.Label(cf,text="Speed:",fg="#ccc",bg="#1a1a2e",
            font=("Arial",9)).pack(side="left",padx=(12,3))
        self.speed_var=tk.IntVar(value=200)
        tk.Scale(cf,from_=50,to=800,orient="horizontal",
            variable=self.speed_var,bg="#1a1a2e",fg="white",
            troughcolor="#333",length=110,highlightthickness=0).pack(side="left")

    def draw_grid_on_canvas(self,canvas,grid,r_drone,c_drone,
                             drone_col,draw_zones=False):
        canvas.delete("all")
        cs=self.cell
        target_cells=set(get_zone_cells(*self.target_zone)) if(
            draw_zones and self.target_zone) else set()
        for r in range(ROWS):
            for c in range(COLS):
                x1,y1=c*cs,r*cs;x2,y2=x1+cs,y1+cs
                mx,my=x1+cs//2,y1+cs//2
                z=grid[(r,c)]
                if z["is_station"]:                          col="#f39c12"
                elif draw_zones and (r,c) in target_cells:
                    col="#003366" if z["covered"]==0 else(
                        "#0a4a2a" if z["covered"]>=100 else "#004488")
                elif z["threat"] and z.get("threat_detected",False): col="#8e44ad"
                elif z["threat"]:                            col="#c0392b"
                elif z["covered"]>=100:                      col="#1abc9c" if draw_zones else "#2980b9"
                elif z["covered"]>0:                         col="#148f77" if draw_zones else "#1a5276"
                else:                                        col="#0d1b2a"
                border="#00aaff" if(draw_zones and (r,c) in target_cells
                    and not z["is_station"]) else "#1a1a2e"
                canvas.create_rectangle(x1,y1,x2,y2,fill=col,
                    outline=border,width=2 if border!="#1a1a2e" else 1)
                if z["is_station"]:
                    canvas.create_text(mx,my,text="⚡",font=("Arial",10))
                elif z["threat"] and z.get("threat_detected",False):
                    canvas.create_text(mx,my,text="✅",font=("Arial",9))
                elif z["threat"]:
                    canvas.create_text(mx,my,text="🚨",font=("Arial",9))
        if draw_zones:
            for zr in range(math.ceil(ROWS/ZONE_SIZE)+1):
                canvas.create_line(0,zr*ZONE_SIZE*cs,COLS*cs,
                    zr*ZONE_SIZE*cs,fill="#334455",dash=(2,4))
            for zc in range(math.ceil(COLS/ZONE_SIZE)+1):
                canvas.create_line(zc*ZONE_SIZE*cs,0,zc*ZONE_SIZE*cs,
                    ROWS*cs,fill="#334455",dash=(2,4))
        dx,dy=c_drone*cs+cs//2,r_drone*cs+cs//2
        canvas.create_oval(dx-10,dy-10,dx+10,dy+10,
            fill=drone_col,outline="#fff",width=2)
        canvas.create_text(dx,dy,text="✈",font=("Arial",10,"bold"),
            fill="#1a1a2e" if drone_col=="#f0c040" else "#fff")

    def draw_grids(self):
        self.draw_grid_on_canvas(self.sc_,self.gs,self.sr,self.sc,
            "#f0c040",draw_zones=True)
        self.draw_grid_on_canvas(self.grec_,self.ggre,self.grer,self.grec,"#d7bde2")
        self.draw_grid_on_canvas(self.gc_,self.ggps,self.gr,self.gc,"#e74c3c")

    def do_step(self):
        if self.step>=300: self.finish();return
        self.step+=1

        # ════════════════════════════════════════════════════
        # SMRS + TWO-HOP ENERGY FEASIBILITY CHECK
        # ════════════════════════════════════════════════════

        # Emergency: battery 0 হলে সরাসরি station
        if self.sb<=0 and not self.going_to_station:
            self.going_to_station=True
            self.station_target=nearest_st(self.sr,self.sc)

        if self.going_to_station:
            # ── Station এ যাওয়ার পথে ──
            if (self.sr,self.sc)==self.station_target:
                # Station এ পৌঁছে গেছি
                self.sb=100; self.sR+=1
                self.going_to_station=False
                self.post_recharge=True
                self.info_lbl.config(
                    text=f"⚡ Step {self.step}: Recharged → Selecting zone...",
                    fg="#ffdd57")
            else:
                # Station এর দিকে যাও — move_toward এ station ঢুকতে দেবে
                nr,nc=move_toward(
                    self.sr,self.sc,
                    self.station_target[0],self.station_target[1])
                # আটকে গেলে সরাসরি station এ
                if nr==self.sr and nc==self.sc:
                    nr,nc=self.station_target
                self.sr,self.sc=nr,nc
                self.sb=max(0,self.sb-ENERGY_PER_CELL)
                # পথেও cell cover করো
                if not self.gs[(self.sr,self.sc)]["is_station"]:
                    self.gs[(self.sr,self.sc)]["covered"]=100
                    self.gs[(self.sr,self.sc)]["last_visited_step"]=self.step

        elif self.post_recharge:
            # ── Recharge এর পরে — Zone priority calculate করো ──
            rankings=rank_zones(
                self.sr,self.sc,self.gs,self.s_detected,self.step)
            best=rankings[0]
            self.target_zone=best["zone"]
            self.target_cell=find_uncovered_in_zone(*best["zone"],self.gs)
            self.post_recharge=False
            self.info_lbl.config(
                text=f"🎯 Zone {best['zone']} (score:{best['score']:.2f})",
                fg="#aaffaa")

        else:
            # ── Normal operation ──

            # Two-hop check: পরের move safe কিনা দেখো
            next_r=self.target_cell[0] if self.target_cell else self.sr
            next_c=self.target_cell[1] if self.target_cell else self.sc
            can_fly,e1,e2,e_needed=two_hop_check(
                self.sr,self.sc,next_r,next_c,self.sb)

            self.hop_lbl.config(
                text=f"Two-Hop: E_hop1={e1} E_hop2={e2} "
                     f"E_needed={e_needed} Battery={self.sb}% "
                     f"{'✅' if can_fly else '❌→Station'}")

            # Must recharge check
            must_rc,e_ret,st=must_recharge_now(self.sr,self.sc,self.sb)

            if must_rc or not can_fly:
                # Station এ যেতে হবে
                self.going_to_station=True
                self.station_target=nearest_st(self.sr,self.sc)
                nr,nc=move_toward(
                    self.sr,self.sc,
                    self.station_target[0],self.station_target[1])
                if nr==self.sr and nc==self.sc:
                    nr,nc=self.station_target
                self.sr,self.sc=nr,nc
                self.sb=max(0,self.sb-ENERGY_PER_CELL)
                if not self.gs[(self.sr,self.sc)]["is_station"]:
                    self.gs[(self.sr,self.sc)]["covered"]=100
                    self.gs[(self.sr,self.sc)]["last_visited_step"]=self.step
                self.info_lbl.config(
                    text=f"⚠️ Step {self.step}: Recharge needed "
                         f"(bat:{self.sb}% e_needed:{e_needed}%)",
                    fg="#ff6b6b")
            else:
                # Zone complete check
                if self.target_zone:
                    zr,zc=self.target_zone
                    if all(self.gs[c]["covered"]>=100
                           for c in get_zone_cells(zr,zc)):
                        rankings=rank_zones(
                            self.sr,self.sc,self.gs,self.s_detected,self.step)
                        best=rankings[0]
                        self.target_zone=best["zone"]
                        self.target_cell=find_uncovered_in_zone(
                            *best["zone"],self.gs)
                        self.info_lbl.config(
                            text=f"✅ Zone done → New: {best['zone']}",
                            fg="#aaffaa")

                if not self.target_zone:
                    rankings=rank_zones(
                        self.sr,self.sc,self.gs,self.s_detected,self.step)
                    self.target_zone=rankings[0]["zone"]
                    self.target_cell=find_uncovered_in_zone(
                        *self.target_zone,self.gs)

                # Navigate or scan
                if (self.target_cell and
                        get_zone_id(self.sr,self.sc)!=self.target_zone):
                    tr,tc_=self.target_cell
                    nr,nc=move_toward(self.sr,self.sc,tr,tc_)
                    self.sr,self.sc=nr,nc
                else:
                    self.sr,self.sc,_=smart_move(
                        self.sr,self.sc,self.s_pr,self.s_pc,
                        self.gs,self.s_detected,self.target_zone)

                # Cell cover করো
                cell=self.gs[(self.sr,self.sc)]
                if not cell["is_station"]:
                    cell["covered"]=100
                    cell["last_visited_step"]=self.step
                    self.sb=max(0,self.sb-ENERGY_PER_CELL)
                    if cell["threat"] and (self.sr,self.sc) not in self.s_detected:
                        self.s_detected.add((self.sr,self.sc))
                        cell["threat_detected"]=True

                if self.target_zone:
                    self.target_cell=find_uncovered_in_zone(
                        *self.target_zone,self.gs)

        self.s_pr,self.s_pc=self.sr,self.sc

        # ════════════════════════════════════════
        # GREEDY (20% threshold baseline)
        # ════════════════════════════════════════
        must_rc_gre, _, _ = must_recharge_now(self.grer, self.grec, self.greb)
        
        if must_rc_gre:
            st=nearest_st(self.grer,self.grec)
            self.grer,self.grec=st; self.greb=100; self.greR+=1
        else:
            self.grer,self.grec=greedy_move(
                self.grer,self.grec,self.gre_pr,self.gre_pc,
                self.ggre,self.gre_detected)
            cell=self.ggre[(self.grer,self.grec)]
            cell["covered"]=100; self.greb=max(0,self.greb-ENERGY_PER_CELL)
            if cell["threat"] and (self.grer,self.grec) not in self.gre_detected:
                self.gre_detected.add((self.grer,self.grec))
                cell["threat_detected"]=True
        self.gre_pr,self.gre_pc=self.grer,self.grec

        # ════════════════════════════════════════
        # GPS SNAKE (20% threshold baseline)
        # ════════════════════════════════════════
        if self.gb<=20:
            gt=nearest_st(self.gr,self.gc)
            self.gr,self.gc=gt; self.gb=100; self.gR+=1
        else:
            self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr)
            self.ggps[(self.gr,self.gc)]["covered"]=100
            self.gb=max(0,self.gb-ENERGY_PER_CELL)

        # Stats
        sc_=round(sum(1 for v in self.gs.values()
            if v["covered"]>=100)/(ROWS*COLS)*100,1)
        grec_=round(sum(1 for v in self.ggre.values()
            if v["covered"]>=100)/(ROWS*COLS)*100,1)
        gc_=round(sum(1 for v in self.ggps.values()
            if v["covered"]>=100)/(ROWS*COLS)*100,1)
        st_=len(self.s_detected)
        gret_=len(self.gre_detected)
        gt_=sum(1 for t in self.threats if self.ggps[t]["covered"]>=100)

        self.log.append({
            "step":self.step,
            "s_cov":sc_,"gre_cov":grec_,"g_cov":gc_,
            "s_bat":self.sb,"gre_bat":self.greb,"g_bat":self.gb,
            "s_thr":st_,"gre_thr":gret_,"g_thr":gt_,
            "s_RC":self.sR,"gre_RC":self.greR,"g_RC":self.gR
        })

        self.step_lbl.config(text=f"Step:{self.step}/300")
        self.s_stat.config(
            text=f"Bat:{self.sb}% | RC:{self.sR} | Cov:{sc_}% | Thr:{st_}/3")
        self.gre_stat.config(
            text=f"Bat:{self.greb}% | RC:{self.greR} | Cov:{grec_}% | Thr:{gret_}/3")
        self.g_stat.config(
            text=f"Bat:{self.gb}% | RC:{self.gR} | Cov:{gc_}% | Thr:{gt_}/3")
        self.draw_grids()

    def toggle_run(self):
        self.running=not self.running
        self.btn_run.config(
            text="⏸ Pause" if self.running else "▶ Start",
            bg="#e67e22" if self.running else "#27ae60")
        if self.running: self.auto_run()

    def auto_run(self):
        if self.running and self.step<300:
            self.do_step()
            self.root.after(self.speed_var.get(),self.auto_run)
        elif self.step>=300:
            self.running=False
            self.btn_run.config(text="▶ Start",bg="#27ae60")
            self.finish()

    def reset(self):
        self.running=False
        self.btn_run.config(text="▶ Start",bg="#27ae60")
        self.reset_state()
        self.info_lbl.config(text="—")
        self.hop_lbl.config(text="Two-Hop: E_hop1=— E_hop2=— E_needed=— Battery=—")
        self.draw_grids()

    def finish(self):
        self.running=False
        last=self.log[-1] if self.log else {}
        try:
            with open('drone_final_results.json','w') as f:
                json.dump(self.log,f,indent=2,default=str)
            saved="💾 drone_final_results.json"
        except Exception as e: saved=str(e)

        msg=(
            f"{'─'*50}\n"
            f"  FINAL RESULTS (300 steps)\n"
            f"{'─'*50}\n"
            f"🧠 SMRS   → Cov:{last.get('s_cov',0)}%  "
            f"Thr:{last.get('s_thr',0)}/3  RC:{self.sR}\n"
            f"🎯 Greedy → Cov:{last.get('gre_cov',0)}%  "
            f"Thr:{last.get('gre_thr',0)}/3  RC:{self.greR}\n"
            f"🛰️  GPS    → Cov:{last.get('g_cov',0)}%  "
            f"Thr:{last.get('g_thr',0)}/3  RC:{self.gR}\n"
            f"{'─'*50}\n"
            f"{saved}"
        )
        messagebox.showinfo("Simulation Complete",msg)

if __name__=="__main__":
    root=tk.Tk()
    app=DroneSimGUI(root)
    root.mainloop()
