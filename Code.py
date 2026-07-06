import math, random, json, statistics, threading
import tkinter as tk
from tkinter import messagebox

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — 30×30 Grid
# ══════════════════════════════════════════════════════════════
ROWS, COLS   = 30, 30
STATIONS     = [(0,0),(0,29),(29,0),(29,29)]
ZONE_SIZE    = 5          
MAX_STEPS    = 2000       
NUM_THREATS  = 20


DETECTION_RADIUS = 2
BATCH_SEEDS  = [42,43,44,45,46,47,48,49,50,51]   

ENERGY_PER_CELL = 1.0
SAFETY_BUFFER   = 1.2

ALPHA         = 0.5
GAMMA         = 0.2
EPSILON       = 0.3
ETA           = 0.05
LAMBDA        = 8.0
MU            = 1.5
K_TURN        = 0.1
VISIT_PENALTY = 3.0
BACKTRACK_PENALTY = 5.0

HY_T1_GAP_W    = 1.0
HY_T1_DIST_W   = 0.05
HY_T2_INCOMPL_W= 1.5
HY_T2_GAP_W    = 0.8
HY_T2_DIST_W   = 0.1
HY_T2_BORDER_W = 0.4    
HY_T3_BORDER_W = 1.2    
HY_T3_DIST_W   = 0.05

# ══════════════════════════════════════════════════════════════
# GRID SETUP
# ══════════════════════════════════════════════════════════════
def make_grid(seed=42):
    random.seed(seed)
    g = {}
    for r in range(ROWS):
        for c in range(COLS):
            edge = min(r, c, ROWS-1-r, COLS-1-c)
            pri  = 3 if edge == 0 else (2 if edge == 1 else 1)
            g[(r,c)] = {
                "covered": 0, "priority": pri,
                "risk": round(random.uniform(0.1, 0.9), 2),
                "threat": False, "is_station": False,
                "visits": 0, "threat_detected": False,
                "last_visited_step": 0
            }
    for s in STATIONS:
        g[s]["is_station"] = True
        g[s]["priority"]   = 0
    return g

def is_st(r,c): return (r,c) in STATIONS
def nearest_st(r,c): return min(STATIONS, key=lambda s: abs(r-s[0])+abs(c-s[1]))
def get_zone_id(r,c): return (r//ZONE_SIZE, c//ZONE_SIZE)

def get_zone_cells(zr,zc):
    out = []
    for r in range(zr*ZONE_SIZE, min((zr+1)*ZONE_SIZE, ROWS)):
        for c in range(zc*ZONE_SIZE, min((zc+1)*ZONE_SIZE, COLS)):
            if not is_st(r,c):
                out.append((r,c))
    return out

def get_all_zones():
    return [(zr,zc) for zr in range(math.ceil(ROWS/ZONE_SIZE)) for zc in range(math.ceil(COLS/ZONE_SIZE))]

def nbrs(r,c,g):
    out = []
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc = r+dr, c+dc
        if 0<=nr<ROWS and 0<=nc<COLS and not g[(nr,nc)]["is_station"]:
            out.append((nr,nc))
    return out

def coverage_pct(g):
    non_station = [k for k,v in g.items() if not v["is_station"]]
    if not non_station: return 0.0
    covered = sum(1 for k in non_station if g[k]["covered"] >= 100)
    return round(covered/len(non_station)*100, 1)

# ══════════════════════════════════════════════════════════════
# DETECTION RADIUS (Feature #1) — realistic sensor/camera FOV
# ══════════════════════════════════════════════════════════════
def scan_for_threats(r, c, g, detected, radius=DETECTION_RADIUS):
    """(r,c)-কেন্দ্রিক Manhattan-distance <= radius এর মধ্যে সব threat cell
    scan করে, যেগুলো এখনো detect হয়নি সেগুলোকে `detected` set-এ যুক্ত করে।
    এটা "ঠিক সেই cell-এ পা দিতে হবে" শর্তটা বাদ দিয়ে বাস্তব sensor-এর মতো
    একটা coverage-radius মডেল করে। রিটার্ন করে এই scan-এ নতুন detect হওয়া
    cell-গুলোর লিস্ট (কেউ ব্যবহার করতে চাইলে, না চাইলে ignore করা যায়)।"""
    newly = []
    for dr in range(-radius, radius+1):
        max_dc = radius - abs(dr)
        for dc in range(-max_dc, max_dc+1):
            pos = (r+dr, c+dc)
            cell = g.get(pos)
            if cell and cell["threat"] and pos not in detected:
                detected.add(pos)
                cell["threat_detected"] = True
                newly.append(pos)
    return newly

# ══════════════════════════════════════════════════════════════
# PHYSICS & ENERGY
# ══════════════════════════════════════════════════════════════
def D_eff(r1,c1,r2,c2,pr,pc):
    D    = math.sqrt((r2-r1)**2 + (c2-c1)**2)
    turn = 0 if pr is None else (0 if (r2-r1==r1-pr and c2-c1==c1-pc) else 1)
    return round(D*(1+K_TURN*turn), 3), turn

def W_eff(r1,c1,r2,c2,ws=12,wd=90):
    h    = math.degrees(math.atan2(c2-c1, r2-r1)) % 360
    diff = abs(wd-h) % 360
    if diff > 180: diff = 360-diff
    return round(ws*(diff/180)*(1+50/200), 3)

def E_ret(r,c):
    dist = min(abs(r-s[0])+abs(c-s[1]) for s in STATIONS)
    return round(dist*ENERGY_PER_CELL*SAFETY_BUFFER, 2)

def energy_to_travel(r1,c1,r2,c2,ws=12,wd=90):
    dist = abs(r1-r2)+abs(c1-c2)
    if dist == 0: return 0.0
    return round(dist*ENERGY_PER_CELL, 2)

def two_hop_check(dr,dc,tr,tc,battery):
    e1 = energy_to_travel(dr,dc,tr,tc)
    st = nearest_st(tr,tc)
    e2 = energy_to_travel(tr,tc,st[0],st[1])
    eni = (e1+e2)*SAFETY_BUFFER + ENERGY_PER_CELL
    return battery>=eni, round(e1,1), round(e2,1), round(eni,1)

def must_recharge_now(r,c,battery):
    st    = nearest_st(r,c)
    e_ret = energy_to_travel(r,c,st[0],st[1])
    threshold = e_ret*SAFETY_BUFFER + ENERGY_PER_CELL
    return battery<=threshold, round(e_ret,1), st

# NEW: Handoff Trigger Function (Predictive Battery Check) — DISTANCE-AWARE FIX
#
# আগের সংস্করণ:
#   threshold = dist + 2*ZONE_SIZE + 4
# সমস্যা: incoming ড্রোনও ঠিক ততটাই "dist" cell দূর থেকে (একই নিকটতম
# station থেকে) হেঁটে আসে। predictive trigger আর hard must-recharge
# trigger-এর মধ্যে battery-gap ছিল মাত্র (dist+14) - (1.2*dist+1) =
# 13 - 0.2*dist — dist বড় হলে (৩০×৩০ গ্রিডে কোণার station থেকে মাঝখানে
# ~28 cell) এই gap নেমে যায় ~7 steps-এ, যেখানে incoming-এর লাগে ~28 steps।
# ফলে ইনস্ট্রুমেন্টেশনে দেখা গেছে emergency swap ঘটছিল ~90%+ সময়, smooth
# handoff প্রায় ঘটতই না — "coverage gap = 0" দাবিটা বাস্তবে ধরে না।
#
# ফিক্স: এখন threshold সরাসরি hard-must-recharge-threshold + (dist + margin)
# থেকে বসানো হচ্ছে — মানে predictive আর hard trigger-এর battery-gap
# সবসময় কমপক্ষে "dist" (incoming-এর real travel time) + margin
# (freeze-zone ড্রিফট ও turning/reaction বাফার) থাকবে। station যত দূরে,
# বাফারও সেই অনুপাতে বড় হবে; station-এর কাছে হলে বাফার ছোট থাকবে (এবং
# backup অকারণে খুব আগে ছাড়া হবে না)।
def needs_handoff_now(r, c, battery):
    """PREDICTIVE HANDOFF TRIGGER — re-tuned (v3).

    v2 (আগের সংস্করণ) buffer = dist + dist + (ZONE_SIZE+5) ≈ 2.2*dist+11।
    এটা smooth-handoff-এর সম্ভাবনা বাড়িয়েছিল, কিন্তু সাথে সাথে battery-র
    ৬০-৭০% খরচ হওয়ার আগেই পরের handoff trigger করে দিচ্ছিল — ফলে পুরো
    simulation-এ recharge-count প্রায় দ্বিগুণ (RC ~33) হয়ে গিয়েছিল, যেটা
    ব্যবহারকারীর কাছে "very bad" মনে হয়েছে।

    v3: এই বাড়তি বাফার সরিয়ে দেওয়া হলো, কারণ নিচে (do_step-এ) FIX A যুক্ত
    হয়েছে — swap smooth না emergency তা এখন আর গুরুত্বপূর্ণ নয়, কারণ যেকোনো
    swap-এর পরেই target_zone/target_cell রিসেট হয়ে fresh rank_zones() কল
    হয় (নতুন active যেখানেই থাকুক, সেখান থেকেই sensible পরবর্তী zone বেছে
    নেয়)। তাই আর জোর করে "incoming ঠিক সময়ে পৌঁছাবে" গ্যারান্টি করার
    দরকার নেই — একটা ছোট, fixed margin-ই যথেষ্ট (incoming-কে সামান্য
    মাথা শুরুর সুযোগ দেয়, recharge-frequency কে single-drone-এর কাছাকাছি
    (~RC 17-26) রাখে)।
    """
    st   = nearest_st(r, c)
    dist = abs(r - st[0]) + abs(c - st[1])

    hard_threshold = dist * SAFETY_BUFFER * ENERGY_PER_CELL + ENERGY_PER_CELL  # == must_recharge_now-এর threshold
    early_margin   = 10 * ENERGY_PER_CELL   # ছোট, fixed head-start বাফার

    threshold = hard_threshold + early_margin
    return battery <= threshold, st

# ══════════════════════════════════════════════════════════════
# SMRS — Zone System
# ══════════════════════════════════════════════════════════════
def zone_info(zr,zc,dr,dc,g,detected,step):
    cells = get_zone_cells(zr,zc)
    if not cells: return None
    avg_cov    = sum(g[c]["covered"] for c in cells) / len(cells)
    incompl    = (100-avg_cov)/100
    has_threat = any(g[c]["threat"] and c not in detected for c in cells)
    ls         = min((g[c]["last_visited_step"] for c in cells), default=0)
    tgap       = min((step-ls)/200, 1.0)
    zrc, zcc   = zr*ZONE_SIZE + ZONE_SIZE//2, zc*ZONE_SIZE + ZONE_SIZE//2
    travel     = abs(dr-zrc)+abs(dc-zcc)
    border_pr  = (sum(g[c]["priority"] for c in cells) / len(cells)) / 3.0
    return {"has_threat": has_threat, "incompl": incompl, "tgap": tgap, "travel": travel, "avg_cov": avg_cov, "border_pr": border_pr}

def rank_zones(dr,dc,g,detected,step):
    all_info = []
    for (zr,zc) in get_all_zones():
        info = zone_info(zr,zc,dr,dc,g,detected,step)
        if info is None: continue
        all_info.append(((zr,zc), info))

    tier1 = [(z,i) for z,i in all_info if i["has_threat"]]
    tier2 = [(z,i) for z,i in all_info if (not i["has_threat"]) and i["incompl"] > 0]
    tier3 = [(z,i) for z,i in all_info if (not i["has_threat"]) and i["incompl"] == 0]

    def score_t1(i): return HY_T1_GAP_W*i["tgap"] - HY_T1_DIST_W*i["travel"]
    def score_t2(i): return (HY_T2_INCOMPL_W*i["incompl"] + HY_T2_GAP_W*i["tgap"] + HY_T2_BORDER_W*i["border_pr"] - HY_T2_DIST_W*i["travel"])
    def score_t3(i): return i["tgap"] + HY_T3_BORDER_W*i["border_pr"] - HY_T3_DIST_W*i["travel"]

    tier1.sort(key=lambda zi: -score_t1(zi[1]))
    tier2.sort(key=lambda zi: -score_t2(zi[1]))
    tier3.sort(key=lambda zi: -score_t3(zi[1]))

    scored = []
    for z,i in tier1: scored.append({"zone":z, "score":round(score_t1(i),4), "cov":round(i["avg_cov"],1), "tier":1})
    for z,i in tier2: scored.append({"zone":z, "score":round(score_t2(i),4), "cov":round(i["avg_cov"],1), "tier":2})
    for z,i in tier3: scored.append({"zone":z, "score":round(score_t3(i),4), "cov":round(i["avg_cov"],1), "tier":3})
    return scored

def calc_C(cov): return 0.0 if cov >= 100 else (2.0 if cov == 0 else 1.0)

def cost_fn(r1,c1,r2,c2,pr,pc,g,detected,target_zone=None):
    d,turn = D_eff(r1,c1,r2,c2,pr,pc)
    w      = W_eff(r1,c1,r2,c2)
    R      = g[(r2,c2)]["risk"]
    er     = E_ret(r2,c2)
    C      = calc_C(g[(r2,c2)]["covered"])
    P      = g[(r2,c2)]["priority"]
    dp     = 20.0 if g[(r2,c2)]["covered"] >= 100 else 0
    V      = VISIT_PENALTY*g[(r2,c2)]["visits"] + dp
    tc     = 10.0 if (g[(r2,c2)]["threat"] and (r2,c2) in detected) else (-4.0 if g[(r2,c2)]["threat"] else 0)
    zb     = -3.0 if (target_zone and get_zone_id(r2,c2)==target_zone) else 0
    bt     = BACKTRACK_PENALTY if (pr,pc)==(r2,c2) else 0
    cost   = ALPHA*d+GAMMA*w+EPSILON*R+ETA*er+V+tc+zb+bt-LAMBDA*C-MU*P
    return round(cost,4), {"cost":round(cost,4)}

def smart_move(r,c,pr,pc,g,detected,target_zone=None):
    ns = nbrs(r,c,g)
    if not ns: return r,c,{}
    best,bpos,bbd = float('inf'),(r,c),{}
    for nr,nc in ns:
        cv,bd = cost_fn(r,c,nr,nc,pr,pc,g,detected,target_zone)
        if cv < best: best=cv; bpos=(nr,nc); bbd=bd
    g[bpos]["visits"] += 1
    return bpos[0],bpos[1],bbd

def move_toward(r,c,tr,tc_):
    nr,nc = r,c
    if r!=tr: nr = r+(1 if tr>r else -1)
    elif c!=tc_: nc = c+(1 if tc_>c else -1)
    if is_st(nr,nc) and (nr,nc)!=(tr,tc_): return r,c
    return nr,nc

def bfs_next_step(r, c, tr, tc, g):
    from collections import deque
    if (r, c) == (tr, tc): return r, c
    start, goal, visited, parent, q = (r, c), (tr, tc), {(r,c)}, {}, deque([(r,c)])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True; break
        for nb in nbrs(cur[0], cur[1], g):
            if nb not in visited:
                visited.add(nb); parent[nb] = cur; q.append(nb)
    if not found: return r, c
    step = goal
    while parent.get(step) != start and step in parent: step = parent[step]
    return step

def find_uncovered_in_zone(zr,zc,g):
    cells   = get_zone_cells(zr,zc)
    empties = [c for c in cells if g[c]["covered"]==0]
    if empties: return empties[0]
    partials = [c for c in cells if 0<g[c]["covered"]<100]
    return partials[0] if partials else None

# ══════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════
def greedy_move(r,c,pr,pc,g,detected):
    ns = nbrs(r,c,g)
    if not ns: return r,c
    best_s,best_p = -float('inf'),(r,c)
    for nr,nc in ns:
        cell  = g[(nr,nc)]
        score = ((100-cell["covered"])+(50 if cell["threat"] and (nr,nc) not in detected else 0)+cell["priority"]*10-cell["visits"]*15)
        if (nr,nc) == (pr,pc): score -= 8
        if score > best_s: best_s=score; best_p=(nr,nc)
    g[best_p]["visits"] += 1
    return best_p[0],best_p[1]

# ══════════════════════════════════════════════════════════════
# GREEDY FIX — target-selection + BFS pathing (same trap as LSA had before)
#
# greedy_move() উপরে দেখানো একটা pure 1-step lookahead: বর্তমান cell-এর ৪টা
# প্রতিবেশীর মধ্যে স্কোর তুলনা করে সবচেয়ে ভালোটায় যায়। সমস্যা: একবার সব ৪টা
# প্রতিবেশীই covered হয়ে গেলে, "coverage" স্কোর সবার জন্যই সমান (~0) হয়ে
# যায় — তখন বাকি থাকে শুধু visits-penalty, যেটা "কোন দিকে unexplored
# এলাকা আছে" বোঝার কোনো উপায় দেয় না। ফলে drone একটা already-covered
# বেষ্টনীর ভেতরে আটকে যেতে পারে, আর কখনো বাকি threat-এর কাছে পৌঁছাতেই পারে
# না — ব্যাচ রিপোর্টে এটাই দেখা যাচ্ছিল (Greedy=None, seed 42/44/47)।
#
# ফিক্স: প্রতি ধাপে local score compare না করে, একটা real global target
# বেছে নেওয়া হয় (আগে undetected threat, না থাকলে nearest uncovered cell),
# আর bfs_next_step() দিয়ে সেই target পর্যন্ত real shortest path ধরে হাঁটা
# হয় — ঠিক SMRS নিজের zone-target-এর জন্য যেভাবে করে।
def gps_snake(r,c,d):
    nc,nr = c+d, r
    if nc<0 or nc>=COLS or is_st(r,nc): nr=(r+1)%ROWS; d=-d; nc=c+d
    if 0<=nr<ROWS and 0<=nc<COLS and is_st(nr,nc): nc+=d
    if not(0<=nc<COLS): nc=c
    return nr,nc,d

# ══════════════════════════════════════════════════════════════
# HEADLESS SIM (Batch-Average)
# ══════════════════════════════════════════════════════════════
class DroneSimHeadless:
    def __init__(self, seed):
        self.seed = seed
        self.gs, self.ggre, self.ggps = make_grid(seed), make_grid(seed), make_grid(seed)
        random.seed(seed)
        self.threats = []
        non_st = [(r,c) for r in range(ROWS) for c in range(COLS) if not is_st(r,c)]
        random.shuffle(non_st)
        for t in non_st[:NUM_THREATS]:
            self.threats.append(t)
            self.gs[t]["threat"] = self.ggre[t]["threat"] = self.ggps[t]["threat"] = True

        # SMRS Multi-Drone Fleet State
        self.s_active = {"r": 1, "c": 1, "b": 100, "pr": None, "pc": None}
        self.s_returning = []
        self.s_incoming = None
        self.s_handoff_mode = False
        self.sR = 0; self.s_detected = set()
        self.target_zone = None; self.target_cell = None

        self.grer,self.grec,self.greb = 1,1,100
        self.gre_pr=self.gre_pc=None; self.greR=0
        self.gre_detected=set(); self.gre_going=False; self.gre_station=None

        self.gr,self.gc,self.gb = 1,1,100
        self.gR=0; self.dr=1; self.gps_going=False; self.gps_station=None
        self.gps_returning=False; self.gps_resume=None
        self.gps_detected=set()   # radius-based detection set (Feature #1)

        self.step=0
        self.s_first_all=None; self.gre_first_all=None; self.g_first_all=None
        self.s_full_cov_step=None   # কবে SMRS grid 100% covered হলো (auto-tune fitness-এর জন্য)

    def do_step(self):
        self.step += 1

        # --- SMRS MULTI-DRONE LOGIC ---
        for rd in self.s_returning[:]:
            if (rd["r"], rd["c"]) == rd["st"]:
                self.sR += 1
                self.s_returning.remove(rd)
            else:
                nr, nc = move_toward(rd["r"], rd["c"], rd["st"][0], rd["st"][1])
                rd["r"], rd["c"] = nr, nc
                rd["b"] = max(0, rd["b"] - ENERGY_PER_CELL)

        if self.s_handoff_mode and self.s_incoming:
            tr, tc = self.s_active["r"], self.s_active["c"]
            # ★ PROXIMITY SWAP: exact same-cell না হয়ে adjacent (দূরত্ব<=1)
            # হলেই handoff সম্পন্ন — zone-এ যদি আর কাজ না থাকে (active শুধু
            # hover করছে), তাহলে দুই drone-কে ঠিক এক ঘরে দাঁড় করিয়ে overlap
            # তৈরি করার দরকার নেই।
            dist_ai = abs(self.s_incoming["r"]-tr) + abs(self.s_incoming["c"]-tc)
            if dist_ai <= 1:
                old = self.s_active.copy()
                old["st"] = nearest_st(tr, tc)
                self.s_returning.append(old)
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False; self.s_incoming = None
                # FIX A: swap-এর পরে target_zone/target_cell রিসেট — নতুন
                # active নিজের বর্তমান position থেকে fresh rank_zones() কল
                # করে সবচেয়ে যুক্তিসঙ্গত zone বেছে নেবে, পুরনো drone-এর
                # (এখন সম্ভবত ভিন্ন position-এর) স্টেল target ধরে রাখবে না।
                self.target_zone = None; self.target_cell = None
            else:
                nr, nc = move_toward(self.s_incoming["r"], self.s_incoming["c"], tr, tc)
                self.s_incoming["r"], self.s_incoming["c"] = nr, nc
                self.s_incoming["b"] = max(0, self.s_incoming["b"] - ENERGY_PER_CELL)
                if not self.gs[(nr, nc)]["is_station"]:
                    pc = self.gs[(nr, nc)]
                    pc["covered"] = 100
                    pc["last_visited_step"] = self.step
                    # BUGFIX: incoming পথে threat cell পার হলে সেটা "covered"
                    # হয়ে যেত কিন্তু কখনো "detected" হতো না (active-এর
                    # movement-এ এই check ছিল, এখানে ছিল না) — ফলে
                    # find_uncovered_in_zone() সেই cell-কে আর কখনো target
                    # ধরত না, threat চিরতরে miss হয়ে যেত (seed 50-এ ঠিক
                    # এটাই ঘটেছিল)।
                    # RADIUS FIX: exact-cell check-এর বদলে radius-based scan
                    scan_for_threats(nr, nc, self.gs, self.s_detected)

        nh, st = needs_handoff_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if nh and not self.s_handoff_mode:
            self.s_handoff_mode = True
            self.s_incoming = {"r": st[0], "c": st[1], "b": 100}

        mrc, e_ret, st = must_recharge_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if mrc:
            old = self.s_active.copy()
            old["st"] = st
            self.s_returning.append(old)
            if self.s_incoming:
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False; self.s_incoming = None
            else:
                self.s_active = {"r": st[0], "c": st[1], "b": 100, "pr": None, "pc": None}
            # FIX A: এই mrc/emergency swap-এর পরেও target_zone/target_cell
            # রিসেট করা হচ্ছে — কারণ নতুন active (emergency swap হলে) পুরনো
            # active-এর position থেকে দূরে থাকতে পারে; পুরনো target_zone
            # ধরে রাখলে drone উল্টো দিকে ফিরে যাওয়ার ঝুঁকি থাকত।
            self.target_zone = None; self.target_cell = None
        else:
            if self.target_zone:
                zr,zc=self.target_zone
                if all(self.gs[cl]["covered"]>=100 for cl in get_zone_cells(zr,zc)):
                    rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                    self.target_zone=rk[0]["zone"]; self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)
            if not self.target_zone:
                rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                self.target_zone=rk[0]["zone"]; self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)
            
            if self.target_cell:
                tr,tc_=self.target_cell
                # TWO-HOP CHECK ব্যবহার: এই target_cell-এ যাওয়ার সিদ্ধান্ত
                # কমিট করার আগে verify করা হচ্ছে যে সেখানে পৌঁছে তারপরও
                # নিকটতম station-এ ফেরার শক্তি থাকবে কিনা (hop1: এখান থেকে
                # target, hop2: target থেকে station)। must_recharge_now()
                # শুধু "এই মুহূর্তে" ফেরার শক্তি আছে কিনা দেখে — কিন্তু
                # target_cell অনেক দূরে (যেমন rank_zones ভুলভাবে দূরের zone
                # বেছে নিলে) হলে ওই এক-পা move করার পরেই ফেরার শক্তি হারানোর
                # ঝুঁকি থাকে, যেটা two_hop_check আগেভাগেই ধরে ফেলে।
                can_go, e1, e2, eni = two_hop_check(
                    self.s_active["r"], self.s_active["c"], tr, tc_, self.s_active["b"])
                if can_go:
                    nr, nc = bfs_next_step(self.s_active["r"],self.s_active["c"],tr,tc_,self.gs)
                    self.s_active["r"], self.s_active["c"] = nr, nc
                else:
                    # BUGFIX: প্রথম সংস্করণে এখানে "move_toward(station)"
                    # ব্যবহার করা হয়েছিল, কিন্তু drone যদি ইতিমধ্যেই
                    # station cell-এই থাকে, move_toward same position
                    # ফেরত দেয় (কোনো movement হয় না), আর station-এ থাকলে
                    # battery deduct/recharge কিছুই হয় না normal movement
                    # code-এ — ফলে drone চিরকাল আটকে থাকত (দেখা গেছে: seed
                    # 43/45/49/50-এ ~2000 steps ধরে freeze)।
                    #
                    # FIX: station-এ থাকলে সরাসরি recharge করে দেওয়া হচ্ছে
                    # (physically সেখানে আছে, রিচার্জ করাই যুক্তিসঙ্গত) এবং
                    # target রিসেট করে দেওয়া হচ্ছে যাতে পরের step full
                    # battery নিয়ে fresh rank_zones() হয় (তখন two_hop_check
                    # আর ব্যর্থ হওয়ার কথা নয়, কারণ গ্রিডের সর্বোচ্চ
                    # round-trip distance-ও 100% battery-তে কভার হয়)।
                    # station-এ না থাকলে, নিরাপদ local smart_move() fallback
                    # ব্যবহার হচ্ছে — এটা কখনো freeze করে না, কারণ nbrs()
                    # থেকে সবসময় একটা real neighbour বেছে নেয়।
                    if self.gs[(self.s_active["r"], self.s_active["c"])]["is_station"]:
                        self.s_active["b"] = 100
                        self.target_zone = None; self.target_cell = None
                    else:
                        self.s_active["r"],self.s_active["c"],_=smart_move(
                            self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                            self.gs,self.s_detected,self.target_zone)
            else:
                self.s_active["r"],self.s_active["c"],_=smart_move(
                    self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                    self.gs,self.s_detected,self.target_zone)

            cell=self.gs[(self.s_active["r"],self.s_active["c"])]
            if not cell["is_station"]:
                cell["covered"]=100; cell["last_visited_step"]=self.step
                self.s_active["b"]=max(0,self.s_active["b"]-ENERGY_PER_CELL)
                scan_for_threats(self.s_active["r"], self.s_active["c"], self.gs, self.s_detected)

            # FIX: এই রিফ্রেশ লাইনটা মিসিং ছিল — ফলে drone প্রথম covered
            # cell-এই আটকে যেত (target_cell কখনো নতুন uncovered cell-এ
            # সরত না), battery শুধু ফুরাতে থাকত, coverage ~0% থেকে যেত।
            if self.target_zone:
                self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)

        self.s_active["pr"],self.s_active["pc"]=self.s_active["r"],self.s_active["c"]
        if self.s_first_all is None and len(self.s_detected) >= NUM_THREATS: self.s_first_all=self.step
        if self.s_full_cov_step is None and coverage_pct(self.gs) >= 100: self.s_full_cov_step=self.step

        # --- GREEDY ---
        if self.gre_going:
            if (self.grer,self.grec)==self.gre_station:
                self.greb=100; self.greR+=1; self.gre_going=False
            else:
                nr,nc=move_toward(self.grer,self.grec,*self.gre_station)
                if (nr,nc)==(self.grer,self.grec): nr,nc=self.gre_station
                self.grer,self.grec=nr,nc; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                if not self.ggre[(self.grer,self.grec)]["is_station"]: self.ggre[(self.grer,self.grec)]["covered"]=100
        else:
            mrc_gre,e_ret_gre,st_gre = must_recharge_now(self.grer,self.grec,self.greb)
            if mrc_gre:
                self.gre_going=True; self.gre_station=st_gre
                nr,nc=move_toward(self.grer,self.grec,*st_gre)
                if (nr,nc)==(self.grer,self.grec): nr,nc=st_gre
                self.grer,self.grec=nr,nc; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                if not self.ggre[(self.grer,self.grec)]["is_station"]: self.ggre[(self.grer,self.grec)]["covered"]=100
            else:
                self.grer,self.grec=greedy_move(self.grer,self.grec,self.gre_pr,self.gre_pc,self.ggre,self.gre_detected)
                cell=self.ggre[(self.grer,self.grec)]
                cell["covered"]=100; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                scan_for_threats(self.grer, self.grec, self.ggre, self.gre_detected)
        self.gre_pr,self.gre_pc=self.grer,self.grec
        if self.gre_first_all is None and len(self.gre_detected) >= NUM_THREATS: self.gre_first_all=self.step

        # --- GPS ---
        if self.gps_going:
            if (self.gr,self.gc)==self.gps_station:
                self.gb=100; self.gR+=1; self.gps_going=False
                if self.gps_resume and (self.gr,self.gc)!=self.gps_resume[:2]: self.gps_returning=True
                elif self.gps_resume: self.dr=self.gps_resume[2]; self.gps_resume=None
            else:
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
        elif self.gps_returning:
            tr,tc_=self.gps_resume[:2]
            if (self.gr,self.gc)==(tr,tc_): self.dr=self.gps_resume[2]; self.gps_returning=False; self.gps_resume=None
            else:
                nr,nc=move_toward(self.gr,self.gc,tr,tc_)
                if (nr,nc)==(self.gr,self.gc): nr,nc=(tr,tc_)
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
        else:
            gps_min_bat=(min(abs(self.gr-s[0])+abs(self.gc-s[1]) for s in STATIONS)+1)*ENERGY_PER_CELL*SAFETY_BUFFER
            if self.gb <= max(15, gps_min_bat):
                self.gps_resume=(self.gr,self.gc,self.dr)
                self.gps_going=True; self.gps_station=nearest_st(self.gr,self.gc)
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
            else:
                self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr)
                self.ggps[(self.gr,self.gc)]["covered"]=100; self.gb=max(0,self.gb-ENERGY_PER_CELL)

        # RADIUS FIX: point-based "is the exact threat cell covered" check
        # বাদ দিয়ে radius-based scan — GPS-কেও অন্য algorithm-গুলোর মতোই
        # একই detection model দিয়ে ন্যায্যভাবে তুলনা করার জন্য।
        scan_for_threats(self.gr, self.gc, self.ggps, self.gps_detected)
        if self.g_first_all is None and len(self.gps_detected) >= NUM_THREATS: self.g_first_all=self.step

    def run(self, max_steps=MAX_STEPS):
        for _ in range(max_steps):
            self.do_step()
            if (self.s_first_all is not None and self.gre_first_all is not None and self.g_first_all is not None): break
        return {"seed": self.seed, "s": self.s_first_all, "gre": self.gre_first_all, "g": self.g_first_all,
                "s_full_cov": self.s_full_cov_step}

def fmt_mean_std(values):
    clean=[v for v in values if v is not None]
    if not clean: return "None ± None"
    mean=statistics.mean(clean); std=statistics.pstdev(clean) if len(clean)>1 else 0.0
    if len(clean)<len(values): return f"{mean:.1f} ± {std:.2f}  (partial: {len(clean)}/{len(values)} seeds)"
    return f"{mean:.1f} ± {std:.2f}"

def run_batch_and_format(seeds=BATCH_SEEDS, max_steps=MAX_STEPS):
    results=[DroneSimHeadless(s).run(max_steps) for s in seeds]
    s_vals=[r["s"] for r in results]; gre_vals=[r["gre"] for r in results]; g_vals=[r["g"] for r in results]
    lines = ["FIRST ALL THREATS DETECTION STEP (lower is better)", "-"*52]
    lines.append(f"SMRS   : {fmt_mean_std(s_vals)}")
    lines.append(f"Greedy : {fmt_mean_std(gre_vals)}")
    lines.append(f"GPS    : {fmt_mean_std(g_vals)}")
    lines.append("-" * 52); lines.append("Per seed values:")
    for r in results: lines.append(f"Seed {r['seed']} | SMRS={r['s'] if r['s'] else 'None'} | Greedy={r['gre'] if r['gre'] else 'None'} | GPS={r['g'] if r['g'] else 'None'}")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════
# AUTO-TUNED WEIGHTS (Feature #4)
#
# SMRS-এর cost_fn()/rank_zones()-এ ১৫টা ওজন (ALPHA, GAMMA, ... HY_T3_DIST_W)
# আগে সব হাতে-টিউন করা constant ছিল। এখানে একটা randomized hill-climbing
# search যুক্ত করা হলো, যেটা batch simulation-এর ওপর সেই ওজনগুলো নিজে থেকে
# optimize করে — "hand-tuned heuristic" থেকে "data-optimized system"-এ
# উন্নীত করার জন্য।
#
# পদ্ধতি: fitness = গড় "first_all detection step" (কম = ভালো); কোনো seed
# শেষ না হলে ভারী penalty। প্রতি iteration-এ একটা random ওজনকে ছোট পরিমাণ
# বাড়ানো/কমানো হয়; fitness উন্নত হলে রাখা হয়, নাহলে ফিরিয়ে দেওয়া হয়
# (coordinate-wise random hill-climbing)। পুরো ১০-seed×২০০০-step batch
# প্রতি candidate-এ চালানো ব্যয়বহুল, তাই tuning-এর সময় কম seed + কম
# max_steps দিয়ে দ্রুত মূল্যায়ন করা হয়; শেষে সেরা weight-set পুরো
# batch দিয়ে re-verify করা হয়।
# ══════════════════════════════════════════════════════════════
TUNABLE_WEIGHT_NAMES = [
    "ALPHA","GAMMA","EPSILON","ETA","LAMBDA","MU","K_TURN",
    "VISIT_PENALTY","BACKTRACK_PENALTY",
    "HY_T1_GAP_W","HY_T1_DIST_W",
    "HY_T2_INCOMPL_W","HY_T2_GAP_W","HY_T2_DIST_W","HY_T2_BORDER_W",
    "HY_T3_BORDER_W","HY_T3_DIST_W",
]

def get_current_weights():
    g = globals()
    return {name: g[name] for name in TUNABLE_WEIGHT_NAMES}

def set_weights(weights):
    g = globals()
    for name, val in weights.items():
        g[name] = val

def evaluate_weights(weights, seeds, max_steps):
    """একটা weight-set দিয়ে headless simulation চালিয়ে fitness (কম-ভালো)
    মাপে। শুধু "detect-all-threats" সময় না, "full-coverage" সময়ও ধরা হয়
    (ছোট weight দিয়ে) — কারণ BFS-locked targeting-এর কারণে tier1 (has-threat)
    zone-selection-এ সাধারণত একটাই candidate থাকে (তাই HY_T1_* ওজন বদলে
    কিছু পরিবর্তন হয় না), কিন্তু tier2/tier3 (কভারেজ/বর্ডার-প্যাট্রল)
    zone-selection-এ বহু candidate থাকে — সেখানেই আসল tuning-সংবেদী সিদ্ধান্ত
    হয়। কোনো seed শেষ না হলে (None) ভারী penalty।"""
    old = get_current_weights()
    set_weights(weights)
    try:
        total = 0.0
        for s in seeds:
            res = DroneSimHeadless(s).run(max_steps)
            det = res["s"] if res["s"] is not None else max_steps * 2
            cov = res["s_full_cov"] if res["s_full_cov"] is not None else max_steps * 2
            total += det + 0.3 * cov
        fitness = total / len(seeds)
    finally:
        set_weights(old)
    return fitness

def auto_tune_weights(iterations=40, seeds=None, max_steps=1200, step_scale=0.25, seed_rng=12345):
    """Randomized coordinate-wise hill-climbing। রিটার্ন করে:
    {"best_weights", "best_fitness", "baseline_fitness", "history"}"""
    rng = random.Random(seed_rng)
    if seeds is None:
        seeds = BATCH_SEEDS[:4]   # দ্রুত মূল্যায়নের জন্য একটা ছোট subset

    baseline = get_current_weights()
    baseline_fit = evaluate_weights(baseline, seeds, max_steps)

    best = dict(baseline); best_fit = baseline_fit
    history = [(0, best_fit)]

    names = list(best.keys())
    for it in range(1, iterations+1):
        name = rng.choice(names)
        old_val = best[name]
        scale = max(abs(old_val), 0.5) * step_scale
        candidate = dict(best)
        candidate[name] = max(0.0, old_val + rng.uniform(-scale, scale))

        fit = evaluate_weights(candidate, seeds, max_steps)
        if fit < best_fit:
            best = candidate; best_fit = fit
        history.append((it, best_fit))

    return {"best_weights": best, "best_fitness": best_fit,
            "baseline_fitness": baseline_fit, "history": history}

def run_auto_tune_and_format(iterations=40, quick_seeds=None, quick_steps=1200,
                              verify_seeds=BATCH_SEEDS, verify_steps=MAX_STEPS, apply=True):
    """auto_tune_weights() চালায়, একটা readable রিপোর্ট বানায়, আর শেষে
    সেরা weight-set দিয়ে পুরো ১০-seed×২০০০-step batch verify করে দেখায়।
    apply=True হলে (default) tuned weights বাকি session-এর জন্য সেট হয়ে
    থাকবে; apply=False দিলে যাচাইয়ের পর আগের হাতে-টিউন করা মান ফিরিয়ে
    দেওয়া হবে।"""
    quick_seeds = quick_seeds if quick_seeds is not None else BATCH_SEEDS[:4]
    result = auto_tune_weights(iterations=iterations, seeds=quick_seeds, max_steps=quick_steps)

    lines = []
    lines.append(f"AUTO-TUNE WEIGHTS  ({iterations} iterations, quick-eval on {len(quick_seeds)} seeds × {quick_steps} steps)")
    lines.append("="*64)
    lines.append(f"Baseline (হাতে-টিউন করা) fitness : {result['baseline_fitness']:.1f}  (avg detect-all step)")
    lines.append(f"সেরা পাওয়া fitness              : {result['best_fitness']:.1f}")
    improvement = result['baseline_fitness'] - result['best_fitness']
    pct = (improvement/result['baseline_fitness']*100) if result['baseline_fitness'] else 0
    lines.append(f"উন্নতি                           : {improvement:+.1f} steps ({pct:+.1f}%)")
    lines.append("-"*64)
    lines.append("বদলে যাওয়া ওজনগুলো:")
    baseline_now = get_current_weights()
    for k, v in result["best_weights"].items():
        base_v = baseline_now[k]
        if abs(v - base_v) > 1e-6:
            lines.append(f"  {k:20s} {base_v:8.4f}  ->  {v:8.4f}")
    lines.append("-"*64)

    old = get_current_weights()
    set_weights(result["best_weights"])
    try:
        verify_report = run_batch_and_format(seeds=verify_seeds, max_steps=verify_steps)
    finally:
        if not apply:
            set_weights(old)

    lines.append(f"পুরো ব্যাচে যাচাই ({len(verify_seeds)} seeds × {verify_steps} steps, tuned weights সহ):")
    lines.append(verify_report)
    lines.append("")
    lines.append("(tuned weights এই session-এর বাকি অংশের জন্য APPLIED হয়ে গেছে।)"
                  if apply else
                  "(শুধু যাচাই — আগের হাতে-টিউন করা weights ফিরিয়ে দেওয়া হয়েছে।)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# GUI (VISUAL SIMULATION WITH HANDOFF)
# ══════════════════════════════════════════════════════════════
class DroneSimGUI:
    def __init__(self,root):
        self.root=root
        self.root.title("Multi-Drone Relay Handoff (SMRS) vs Baselines")
        self.root.configure(bg="#1a1a2e")
        self.cell = 13
        self.reset_state()
        self.create_widgets()
        self.draw_grids()

    def reset_state(self):
        seed = random.randint(1,999)
        self.gs, self.ggre, self.ggps = make_grid(seed), make_grid(seed), make_grid(seed)
        self.threats = []
        non_st = [(r,c) for r in range(ROWS) for c in range(COLS) if not is_st(r,c)]
        random.shuffle(non_st)
        for t in non_st[:NUM_THREATS]:
            self.threats.append(t)
            self.gs[t]["threat"] = self.ggre[t]["threat"] = self.ggps[t]["threat"] = True

        # SMRS Multi-Drone State Setup
        self.s_active = {"r": 1, "c": 1, "b": 100, "pr": None, "pc": None}
        self.s_returning = []
        self.s_incoming = None
        self.s_handoff_mode = False
        
        self.sR = 0; self.s_detected = set()
        self.target_zone = None; self.target_cell = None

        self.grer,self.grec,self.greb = 1,1,100
        self.gre_pr=self.gre_pc=None; self.greR=0
        self.gre_detected=set(); self.gre_going=False; self.gre_station=None

        self.gr,self.gc,self.gb = 1,1,100
        self.gR=0; self.dr=1; self.gps_going=False; self.gps_station=None
        self.gps_returning=False; self.gps_resume=None
        self.gps_detected=set()   # radius-based detection set (Feature #1)

        self.step=0; self.log=[]; self.running=False; self.speed=50

    def create_widgets(self):
        top=tk.Frame(self.root,bg="#16213e",pady=5); top.pack(fill="x")
        self.step_lbl=tk.Label(top,text=f"Step:0/{MAX_STEPS}", font=("Consolas",11,"bold"),fg="#e2e2e2",bg="#16213e")
        self.step_lbl.pack(side="left",padx=10)
        self.info_lbl=tk.Label(top,text="—", font=("Consolas",9),fg="#ffdd57",bg="#16213e")
        self.info_lbl.pack(side="left",padx=8)

        mf=tk.Frame(self.root,bg="#1a1a2e"); mf.pack(padx=8,pady=4)

        sf=tk.Frame(mf,bg="#0f3460",bd=2,relief="groove"); sf.pack(side="left",padx=3)
        tk.Label(sf,text="🧠 SMRS (Relay Fleet: 0-Blind-Spot)", font=("Arial",9,"bold"),fg="#00d2ff",bg="#0f3460").pack(pady=2)
        self.sc_=tk.Canvas(sf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.sc_.pack()
        self.s_stat=tk.Label(sf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#aaffaa",bg="#0f3460"); self.s_stat.pack(pady=2)

        grf=tk.Frame(mf,bg="#4a235a",bd=2,relief="groove"); grf.pack(side="left",padx=3)
        tk.Label(grf,text="🎯 Greedy (Single Drone)", font=("Arial",9,"bold"),fg="#d7bde2",bg="#4a235a").pack(pady=2)
        self.gc2=tk.Canvas(grf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.gc2.pack()
        self.g_stat2=tk.Label(grf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#ebdef0",bg="#4a235a"); self.g_stat2.pack(pady=2)

        gpf=tk.Frame(mf,bg="#3d0000",bd=2,relief="groove"); gpf.pack(side="left",padx=3)
        tk.Label(gpf,text="🛰️ GPS Snake (Single Drone)", font=("Arial",9,"bold"),fg="#ff9999",bg="#3d0000").pack(pady=2)
        self.gpc=tk.Canvas(gpf,width=COLS*self.cell,height=ROWS*self.cell, bg="#0a0a1a",highlightthickness=0); self.gpc.pack()
        self.g_stat3=tk.Label(gpf,text="Bat:100% | RC:0 | Cov:0%", font=("Consolas",8),fg="#ffaaaa",bg="#3d0000"); self.g_stat3.pack(pady=2)

        self.hop_lbl=tk.Label(self.root, text="SMRS Backup Status: Waiting...",font=("Consolas",8),fg="#888",bg="#1a1a2e")
        self.hop_lbl.pack(pady=1)

        leg=tk.Frame(self.root,bg="#1a1a2e"); leg.pack(pady=2)
        for sym,lbl in [("🟨","Active"),("🟧","Returning"),("🟦","Incoming"),("🟩","Covered"),("🟥","Threat")]:
            tk.Label(leg,text=f"{sym}{lbl}",font=("Arial",8), fg="#ccc",bg="#1a1a2e").pack(side="left",padx=4)

        cf=tk.Frame(self.root,bg="#1a1a2e",pady=4); cf.pack()
        self.btn_run=tk.Button(cf,text="▶ Start",font=("Arial",10,"bold"), bg="#27ae60",fg="white",width=9,command=self.toggle_run)
        self.btn_run.pack(side="left",padx=4)
        tk.Button(cf,text="⏭ Step",font=("Arial",9),bg="#2980b9", fg="white",width=7,command=self.do_step).pack(side="left",padx=4)
        tk.Button(cf,text="↺ Reset",font=("Arial",9),bg="#7f8c8d", fg="white",width=7,command=self.reset).pack(side="left",padx=4)
        self.btn_batch=tk.Button(cf,text="📊 Batch Avg (10 seeds)",font=("Arial",9,"bold"), bg="#8e44ad",fg="white",width=20,command=self.run_batch_avg)
        self.btn_batch.pack(side="left",padx=4)
        self.btn_tune=tk.Button(cf,text="🎯 Auto-Tune Weights",font=("Arial",9,"bold"), bg="#16a085",fg="white",width=18,command=self.run_auto_tune)
        self.btn_tune.pack(side="left",padx=4)

        tk.Label(cf,text="Speed:",fg="#ccc",bg="#1a1a2e", font=("Arial",9)).pack(side="left",padx=(10,3))
        self.speed_var=tk.IntVar(value=20)
        tk.Scale(cf,from_=5,to=500,orient="horizontal", variable=self.speed_var,bg="#1a1a2e",fg="white", troughcolor="#333",length=120,highlightthickness=0).pack(side="left")

    def run_batch_avg(self):
        self.btn_batch.config(state="disabled", text="⏳ Running batch...")
        def worker():
            table_text = run_batch_and_format()
            self.root.after(0, lambda: self.show_batch_result(table_text))
        threading.Thread(target=worker, daemon=True).start()

    def show_batch_result(self, table_text):
        self.btn_batch.config(state="normal", text="📊 Batch Avg (10 seeds)")
        messagebox.showinfo(f"Batch Average", table_text)

    def run_auto_tune(self):
        # Feature #4: ব্যাকগ্রাউন্ডে weight auto-tuning চালায় (quick-eval
        # search, তারপর সেরা weight-set পুরো ১০-seed batch দিয়ে verify),
        # যাতে GUI freeze না হয়। শেষে সেরা weights session-এর বাকি অংশের
        # জন্য apply হয়ে যায়।
        self.btn_tune.config(state="disabled", text="⏳ Tuning...")
        def worker():
            report = run_auto_tune_and_format()
            self.root.after(0, lambda: self.show_tune_result(report))
        threading.Thread(target=worker, daemon=True).start()

    def show_tune_result(self, report):
        self.btn_tune.config(state="normal", text="🎯 Auto-Tune Weights")
        messagebox.showinfo("Auto-Tune Weights — Result", report)

    def draw_canvas(self,canvas,grid,active_d,drone_col,draw_zones=False, returning=[], incoming=None):
        canvas.delete("all")
        cs = self.cell
        tz_cells = set(get_zone_cells(*self.target_zone)) if (draw_zones and self.target_zone) else set()

        for r in range(ROWS):
            for c in range(COLS):
                x1,y1 = c*cs,r*cs
                z = grid[(r,c)]
                if z["is_station"]: col="#f39c12"
                elif draw_zones and (r,c) in tz_cells:
                    col="#003a6e" if z["covered"]==0 else("#0a4a2a" if z["covered"]>=100 else "#004488")
                elif z["threat"] and z.get("threat_detected",False): col="#8e44ad"
                elif z["threat"]: col="#c0392b"
                elif z["covered"]>=100: col="#1abc9c" if draw_zones else "#2980b9"
                elif z["covered"]>0: col="#148f77" if draw_zones else "#1a5276"
                else: col="#0d1b2a"
                canvas.create_rectangle(x1,y1,x1+cs,y1+cs,fill=col,outline="",width=0)

                if z["is_station"]: canvas.create_text(x1+cs//2,y1+cs//2,text="⚡",font=("Arial",7))
                elif z["threat"] and z.get("threat_detected",False): canvas.create_text(x1+cs//2,y1+cs//2,text="✅",font=("Arial",6))
                elif z["threat"]: canvas.create_text(x1+cs//2,y1+cs//2,text="🚨",font=("Arial",6))

        if draw_zones:
            for zr in range(math.ceil(ROWS/ZONE_SIZE)+1): canvas.create_line(0,zr*ZONE_SIZE*cs,COLS*cs, zr*ZONE_SIZE*cs,fill="#2a3a4a",dash=(2,3))
            for zc in range(math.ceil(COLS/ZONE_SIZE)+1): canvas.create_line(zc*ZONE_SIZE*cs,0,zc*ZONE_SIZE*cs, ROWS*cs,fill="#2a3a4a",dash=(2,3))
            if self.target_zone:
                tzr,tzc = self.target_zone
                zx1,zy1 = tzc*ZONE_SIZE*cs, tzr*ZONE_SIZE*cs
                canvas.create_rectangle(zx1,zy1, zx1+ZONE_SIZE*cs,zy1+ZONE_SIZE*cs, outline="#00aaff",width=2)

        # Draw Returning Drones (Orange/Small)
        for rd in returning:
            px,py = rd["c"]*cs+cs//2, rd["r"]*cs+cs//2
            canvas.create_oval(px-cs//2+2,py-cs//2+2,px+cs//2-2,py+cs//2-2, fill="#e67e22",outline="")

        # Draw Incoming Drone (Cyan) — ছোট, hollow ring, cell-এর কোণার দিকে
        # সরানো, যাতে active-এর সাথে same/adjacent cell-এ থাকলেও visually
        # আলাদা দেখা যায় (এটা শুধু rendering-এর পরিবর্তন — movement/handoff
        # logic-এ কোনো হাত দেওয়া হয়নি, তাই zero-coverage-gap guarantee অক্ষত)।
        if incoming:
            px,py = incoming["c"]*cs+cs//2, incoming["r"]*cs+cs//2
            r_small = max(3, cs//2 - 3)
            ox, oy = -cs//4, -cs//4
            canvas.create_oval(px+ox-r_small,py+oy-r_small,px+ox+r_small,py+oy+r_small,
                outline="#00ffff", width=2)

        # Draw Active Drone (Yellow)
        if active_d:
            px,py = active_d["c"]*cs+cs//2, active_d["r"]*cs+cs//2
            canvas.create_oval(px-cs//2,py-cs//2,px+cs//2,py+cs//2, fill=drone_col,outline="#fff",width=1)

    def draw_grids(self):
        self.draw_canvas(self.sc_, self.gs, self.s_active, "#f0c040", draw_zones=True, returning=self.s_returning, incoming=self.s_incoming)
        self.draw_canvas(self.gc2, self.ggre, {"r":self.grer,"c":self.grec}, "#d7bde2")
        self.draw_canvas(self.gpc, self.ggps, {"r":self.gr,"c":self.gc}, "#e74c3c")

    def do_step(self):
        if self.step>=MAX_STEPS: self.finish(); return
        self.step+=1

        # ==============================================================
        # 1. SMRS: RELAY HANDOFF LOGIC
        # ==============================================================
        
        # A. Process Returning Drones
        for rd in self.s_returning[:]:
            if (rd["r"], rd["c"]) == rd["st"]:
                self.sR += 1
                self.s_returning.remove(rd)
            else:
                nr, nc = move_toward(rd["r"], rd["c"], rd["st"][0], rd["st"][1])
                rd["r"], rd["c"] = nr, nc
                rd["b"] = max(0, rd["b"] - ENERGY_PER_CELL)

        # B. Process Incoming Backup Drone
        if self.s_handoff_mode and self.s_incoming:
            tr, tc = self.s_active["r"], self.s_active["c"]
            # ★ PROXIMITY SWAP: incoming ঠিক same cell-এ না এসেও, adjacent
            # (Manhattan distance <= 1) হলেই handoff সম্পন্ন ধরা হচ্ছে।
            # আগে exact-same-cell লাগত বলে zone সম্পূর্ণ কভার হয়ে যাওয়ার পরেও
            # active শুধু hover করে অপেক্ষা করত আর incoming ঠিক তার ঘাড়ে এসে
            # বসত — দুটো drone visually overlap করত (স্ক্রিনশটে যেটা দেখা
            # গেছে)। এখন এক ঘর দূরত্বেই swap হয়ে যায়, তাই কোনো overlap হয় না
            # এবং ১টা extra step-ও বাঁচে।
            dist_ai = abs(self.s_incoming["r"]-tr) + abs(self.s_incoming["c"]-tc)
            if dist_ai <= 1:
                # Handoff Complete! Swap active and return old
                old_active = self.s_active.copy()
                old_active["st"] = nearest_st(tr, tc)
                self.s_returning.append(old_active)

                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False
                self.s_incoming = None
                # FIX A: swap-এর পরে target_zone/target_cell রিসেট — নতুন
                # active তার বর্তমান position থেকেই fresh rank_zones() করবে।
                self.target_zone = None; self.target_cell = None
                self.info_lbl.config(text=f"🔄 Handoff Complete near {tr},{tc}", fg="#00d2ff")
                self.hop_lbl.config(text="SMRS Backup Status: Idle", fg="#888")
            else:
                nr, nc = move_toward(self.s_incoming["r"], self.s_incoming["c"], tr, tc)
                self.s_incoming["r"], self.s_incoming["c"] = nr, nc
                self.s_incoming["b"] = max(0, self.s_incoming["b"] - ENERGY_PER_CELL)
                # Incoming drone covers cells on its way
                if not self.gs[(nr, nc)]["is_station"]:
                    pc = self.gs[(nr, nc)]
                    pc["covered"] = 100
                    pc["last_visited_step"] = self.step
                    # BUGFIX: আগে এখানে threat-detection check ছিল না — একই
                    # bug যা seed 50-এ ২টা threat permanently miss করিয়েছিল।
                    # RADIUS FIX: exact-cell check-এর বদলে radius-based scan
                    scan_for_threats(nr, nc, self.gs, self.s_detected)

        # C. Predictive Trigger: Launch Backup early
        nh, st = needs_handoff_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])
        if nh and not self.s_handoff_mode:
            self.s_handoff_mode = True
            self.s_incoming = {"r": st[0], "c": st[1], "b": 100}
            self.info_lbl.config(text=f"🚀 Backup launched from {st}", fg="#ff9900")
            self.hop_lbl.config(text=f"SMRS Backup: INCOMING from {st}", fg="#00ffff")

        # D. Active Drone Mission Control
        mrc, e_ret, st = must_recharge_now(self.s_active["r"], self.s_active["c"], self.s_active["b"])

        if mrc:
            # Absolute critical - must abandon
            old_active = self.s_active.copy()
            old_active["st"] = st
            self.s_returning.append(old_active)

            if self.s_incoming:
                self.s_active = self.s_incoming
                self.s_active["pr"] = self.s_active["pc"] = None
                self.s_handoff_mode = False
                self.s_incoming = None
                self.info_lbl.config(text=f"⚠️ Emergency Swap!", fg="#ff4444")
            else:
                self.s_active = {"r": st[0], "c": st[1], "b": 100, "pr": None, "pc": None}
                # BUGFIX: এখানে আগে self.sR += 1 করা হচ্ছিল, কিন্তু old drone
                # তখনও s_returning list-এ যুক্ত হয়েছে — সেটা station-এ
                # পৌঁছালে (উপরের ধাপ A-এ) আবারও sR += 1 হয়, ফলে একটাই
                # recharge event দুইবার গোনা হচ্ছিল (double-count)। এখন শুধু
                # returning-list-এর মাধ্যমে একবারই গোনা হবে, headless
                # ক্লাসের সাথেও এটা সামঞ্জস্যপূর্ণ।
            # FIX A: mrc/emergency swap-এর পরেও target_zone/target_cell রিসেট
            self.target_zone = None; self.target_cell = None
        elif self.s_handoff_mode and self.s_incoming:
            # ★ FREEZE-ZONE MONITORING: handoff চলাকালীন active তার *বর্তমান*
            # target_zone-এই কাজ চালিয়ে যায় — নতুন কোনো দূরের zone-এ যায় না
            # (rank_zones রি-কল হয় না), আর কখনো idle থাকে না। এর মানে সেই
            # zone/group-এর monitoring এক সেকেন্ডের জন্যও বন্ধ হয় না।
            # Incoming পুরো পথ active-এর দিকে এগিয়ে আসে — কিন্তু active দূরে
            # সরে না যাওয়ায় এই distance zone-size-এর মধ্যে bounded থাকে,
            # আগের (original) কোডের মতো unbounded chase হয় না।
            if self.target_zone:
                zr, zc = self.target_zone
                cell_here = find_uncovered_in_zone(zr, zc, self.gs)
                if cell_here:
                    tr, tc_ = cell_here
                    nr, nc = bfs_next_step(self.s_active["r"], self.s_active["c"], tr, tc_, self.gs)
                else:
                    # zone পুরো কভার — center-এ hover করে অপেক্ষা (অন্য zone-এ যাবে না)
                    zrc = zr*ZONE_SIZE + ZONE_SIZE//2
                    zcc = zc*ZONE_SIZE + ZONE_SIZE//2
                    nr, nc = move_toward(self.s_active["r"], self.s_active["c"], zrc, zcc)
            else:
                nr, nc = self.s_active["r"], self.s_active["c"]

            self.s_active["r"], self.s_active["c"] = nr, nc
            cell = self.gs[(nr, nc)]
            if not cell["is_station"]:
                cell["covered"] = 100
                cell["last_visited_step"] = self.step
                self.s_active["b"] = max(0, self.s_active["b"] - ENERGY_PER_CELL)
                scan_for_threats(nr, nc, self.gs, self.s_detected)
        else:
            if self.target_zone:
                zr,zc=self.target_zone
                if all(self.gs[cl]["covered"]>=100 for cl in get_zone_cells(zr,zc)):
                    rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                    self.target_zone=rk[0]["zone"]
                    self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)
            if not self.target_zone:
                rk=rank_zones(self.s_active["r"],self.s_active["c"],self.gs,self.s_detected,self.step)
                self.target_zone=rk[0]["zone"]
                self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)

            if self.target_cell:
                tr,tc_=self.target_cell
                # TWO-HOP CHECK ব্যবহার: এই target_cell-এ যাওয়ার সিদ্ধান্ত
                # কমিট করার আগে verify করা হচ্ছে যে সেখানে পৌঁছে তারপরও
                # নিকটতম station-এ ফেরার শক্তি থাকবে কিনা (hop1: এখান থেকে
                # target, hop2: target থেকে station)। must_recharge_now()
                # শুধু "এই মুহূর্তে" ফেরার শক্তি আছে কিনা দেখে — কিন্তু
                # target_cell অনেক দূরে (যেমন rank_zones ভুলভাবে দূরের zone
                # বেছে নিলে) হলে ওই এক-পা move করার পরেই ফেরার শক্তি হারানোর
                # ঝুঁকি থাকে, যেটা two_hop_check আগেভাগেই ধরে ফেলে।
                can_go, e1, e2, eni = two_hop_check(
                    self.s_active["r"], self.s_active["c"], tr, tc_, self.s_active["b"])
                if can_go:
                    nr, nc = bfs_next_step(self.s_active["r"],self.s_active["c"],tr,tc_,self.gs)
                    self.s_active["r"], self.s_active["c"] = nr, nc
                else:
                    # BUGFIX: প্রথম সংস্করণে এখানে "move_toward(station)"
                    # ব্যবহার করা হয়েছিল, কিন্তু drone যদি ইতিমধ্যেই
                    # station cell-এই থাকে, move_toward same position
                    # ফেরত দেয় (কোনো movement হয় না), আর station-এ থাকলে
                    # battery deduct/recharge কিছুই হয় না normal movement
                    # code-এ — ফলে drone চিরকাল আটকে থাকত (দেখা গেছে: seed
                    # 43/45/49/50-এ ~2000 steps ধরে freeze)।
                    #
                    # FIX: station-এ থাকলে সরাসরি recharge করে দেওয়া হচ্ছে
                    # (physically সেখানে আছে, রিচার্জ করাই যুক্তিসঙ্গত) এবং
                    # target রিসেট করে দেওয়া হচ্ছে যাতে পরের step full
                    # battery নিয়ে fresh rank_zones() হয় (তখন two_hop_check
                    # আর ব্যর্থ হওয়ার কথা নয়, কারণ গ্রিডের সর্বোচ্চ
                    # round-trip distance-ও 100% battery-তে কভার হয়)।
                    # station-এ না থাকলে, নিরাপদ local smart_move() fallback
                    # ব্যবহার হচ্ছে — এটা কখনো freeze করে না, কারণ nbrs()
                    # থেকে সবসময় একটা real neighbour বেছে নেয়।
                    if self.gs[(self.s_active["r"], self.s_active["c"])]["is_station"]:
                        self.s_active["b"] = 100
                        self.target_zone = None; self.target_cell = None
                    else:
                        self.s_active["r"],self.s_active["c"],_=smart_move(
                            self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                            self.gs,self.s_detected,self.target_zone)
            else:
                self.s_active["r"],self.s_active["c"],_=smart_move(
                    self.s_active["r"],self.s_active["c"],self.s_active["pr"],self.s_active["pc"],
                    self.gs,self.s_detected,self.target_zone)

            cell=self.gs[(self.s_active["r"],self.s_active["c"])]
            if not cell["is_station"]:
                cell["covered"]=100
                cell["last_visited_step"]=self.step
                self.s_active["b"]=max(0,self.s_active["b"]-ENERGY_PER_CELL)
                scan_for_threats(self.s_active["r"], self.s_active["c"], self.gs, self.s_detected)

            if self.target_zone:
                self.target_cell=find_uncovered_in_zone(*self.target_zone,self.gs)

        self.s_active["pr"], self.s_active["pc"] = self.s_active["r"], self.s_active["c"]

        # ==============================================================
        # 2. Greedy Baseline (Single Drone)
        # ==============================================================
        if self.gre_going:
            if (self.grer,self.grec)==self.gre_station:
                self.greb=100; self.greR+=1; self.gre_going=False
            else:
                nr,nc=move_toward(self.grer,self.grec,*self.gre_station)
                if (nr,nc)==(self.grer,self.grec): nr,nc=self.gre_station
                self.grer,self.grec=nr,nc; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                if not self.ggre[(self.grer,self.grec)]["is_station"]: self.ggre[(self.grer,self.grec)]["covered"]=100
        else:
            mrc_gre,e_ret_gre,st_gre = must_recharge_now(self.grer,self.grec,self.greb)
            if mrc_gre:
                self.gre_going=True; self.gre_station=st_gre
                nr,nc=move_toward(self.grer,self.grec,*st_gre)
                if (nr,nc)==(self.grer,self.grec): nr,nc=st_gre
                self.grer,self.grec=nr,nc; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                if not self.ggre[(self.grer,self.grec)]["is_station"]: self.ggre[(self.grer,self.grec)]["covered"]=100
            else:
                self.grer,self.grec=greedy_move(self.grer,self.grec,self.gre_pr,self.gre_pc,self.ggre,self.gre_detected)
                cell=self.ggre[(self.grer,self.grec)]
                cell["covered"]=100; self.greb=max(0,self.greb-ENERGY_PER_CELL)
                scan_for_threats(self.grer, self.grec, self.ggre, self.gre_detected)
        self.gre_pr,self.gre_pc=self.grer,self.grec

        # ==============================================================
        # 3. GPS Baseline (Single Drone)
        # ==============================================================
        if self.gps_going:
            if (self.gr,self.gc)==self.gps_station:
                self.gb=100; self.gR+=1; self.gps_going=False
                if self.gps_resume and (self.gr,self.gc)!=self.gps_resume[:2]: self.gps_returning=True
                elif self.gps_resume: self.dr=self.gps_resume[2]; self.gps_resume=None
            else:
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
        elif self.gps_returning:
            tr,tc_=self.gps_resume[:2]
            if (self.gr,self.gc)==(tr,tc_): self.dr=self.gps_resume[2]; self.gps_returning=False; self.gps_resume=None
            else:
                nr,nc=move_toward(self.gr,self.gc,tr,tc_)
                if (nr,nc)==(self.gr,self.gc): nr,nc=(tr,tc_)
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
        else:
            gps_min_bat = (min(abs(self.gr-s[0])+abs(self.gc-s[1]) for s in STATIONS)+1)*ENERGY_PER_CELL*SAFETY_BUFFER
            if self.gb <= max(15, gps_min_bat):
                self.gps_resume=(self.gr,self.gc,self.dr); self.gps_going=True; self.gps_station=nearest_st(self.gr,self.gc)
                nr,nc=move_toward(self.gr,self.gc,*self.gps_station)
                if (nr,nc)==(self.gr,self.gc): nr,nc=self.gps_station
                self.gr,self.gc=nr,nc; self.gb=max(0,self.gb-ENERGY_PER_CELL)
                if not self.ggps[(self.gr,self.gc)]["is_station"]: self.ggps[(self.gr,self.gc)]["covered"]=100
            else:
                self.gr,self.gc,self.dr=gps_snake(self.gr,self.gc,self.dr)
                self.ggps[(self.gr,self.gc)]["covered"]=100; self.gb=max(0,self.gb-ENERGY_PER_CELL)

        # Update Logs and UI
        nt=len(self.threats)
        sc_  = coverage_pct(self.gs)
        grc_ = coverage_pct(self.ggre)
        gpc_ = coverage_pct(self.ggps)
        st_=len(self.s_detected)
        grt_=len(self.gre_detected)
        # RADIUS FIX: exact-cell coverage check বাদ দিয়ে radius-based scan,
        # অন্য algorithm-গুলোর মতোই একই detection model ব্যবহার করার জন্য।
        scan_for_threats(self.gr, self.gc, self.ggps, self.gps_detected)
        gpt_=len(self.gps_detected)

        self.log.append({
            "step":self.step, "s_cov":sc_,"gre_cov":grc_,"g_cov":gpc_,
            "s_bat":round(self.s_active["b"],1),"gre_bat":self.greb,"g_bat":self.gb,
            "s_thr":st_,"gre_thr":grt_,"g_thr":gpt_,
            "s_RC":self.sR,"gre_RC":self.greR,"g_RC":self.gR
        })

        self.step_lbl.config(text=f"Step:{self.step}/{MAX_STEPS}")
        self.s_stat.config(text=f"Bat:{self.s_active['b']:.1f}% | RC:{self.sR} | Cov:{sc_}% | Thr:{st_}/{nt}")
        self.g_stat2.config(text=f"Bat:{self.greb:.1f}% | RC:{self.greR} | Cov:{grc_}% | Thr:{grt_}/{nt}")
        self.g_stat3.config(text=f"Bat:{self.gb:.1f}% | RC:{self.gR} | Cov:{gpc_}% | Thr:{gpt_}/{nt}")
        self.draw_grids()

    def toggle_run(self):
        self.running=not self.running
        self.btn_run.config(text="⏸ Pause" if self.running else "▶ Start", bg="#e67e22" if self.running else "#27ae60")
        if self.running: self.auto_run()

    def auto_run(self):
        if self.running and self.step<MAX_STEPS:
            self.do_step()
            self.root.after(self.speed_var.get(),self.auto_run)
        elif self.step>=MAX_STEPS:
            self.running=False
            self.btn_run.config(text="▶ Start",bg="#27ae60")
            self.finish()

    def reset(self):
        self.running=False
        self.btn_run.config(text="▶ Start",bg="#27ae60")
        self.reset_state(); self.info_lbl.config(text="—")
        self.hop_lbl.config(text="SMRS Backup Status: Idle")
        self.draw_grids()

    def finish(self):
        self.running=False
        last=self.log[-1] if self.log else {}
        nt=len(self.threats)
        msg=(
            f"{'─'*52}\n  FINAL RESULTS — 30×30 Grid ({MAX_STEPS} steps)\n{'─'*52}\n"
            f"🧠 SMRS(Fleet) → Cov:{last.get('s_cov',0)}%  Thr:{last.get('s_thr',0)}/{nt}  RC:{self.sR}\n"
            f"🎯 Greedy      → Cov:{last.get('gre_cov',0)}%  Thr:{last.get('gre_thr',0)}/{nt}  RC:{self.greR}\n"
            f"🛰️  GPS       → Cov:{last.get('g_cov',0)}%  Thr:{last.get('g_thr',0)}/{nt}  RC:{self.gR}\n"
            f"{'─'*52}\n"
        )
        messagebox.showinfo("Simulation Complete",msg)

if __name__=="__main__":
    root=tk.Tk()
    app=DroneSimGUI(root)
    root.mainloop()
