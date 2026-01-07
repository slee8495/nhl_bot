from __future__ import annotations
import re

# NHL canonical 32 team abbreviations (3-letter)
ABBR = {
    "ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA",
    "LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS",
    "STL","TBL","TOR","VAN","VGK","WPG","WSH","UTA"
}

ALIASES = {
    # --- ANA (Ducks) ---
    "ANA":"ANA","ANH":"ANA","ANAHEIM":"ANA","DUCKS":"ANA","ANAHEIM DUCKS":"ANA",

    # --- BOS (Bruins) ---
    "BOS":"BOS","BOSTON":"BOS","BRUINS":"BOS","BOSTON BRUINS":"BOS",

    # --- BUF (Sabres) ---
    "BUF":"BUF","BUFFALO":"BUF","SABRES":"BUF","SABERS":"BUF","BUFFALO SABRES":"BUF",

    # --- CAR (Hurricanes) ---
    "CAR":"CAR","CAROLINA":"CAR","HURRICANES":"CAR","CANES":"CAR","CAROLINA HURRICANES":"CAR",

    # --- CBJ (Blue Jackets) ---
    "CBJ":"CBJ","CLB":"CBJ","COLUMBUS":"CBJ","BLUE JACKETS":"CBJ","BLUEJACKETS":"CBJ",
    "COLUMBUS BLUE JACKETS":"CBJ","COLUMBUS BLUEJACKETS":"CBJ",

    # --- CGY (Flames) ---
    "CGY":"CGY","CALGARY":"CGY","FLAMES":"CGY","CALGARY FLAMES":"CGY",

    # --- CHI (Blackhawks) ---
    "CHI":"CHI","CHICAGO":"CHI","BLACKHAWKS":"CHI","BLACK HAWKS":"CHI","CHICAGO BLACKHAWKS":"CHI",

    # --- COL (Avalanche) ---
    "COL":"COL","COLORADO":"COL","AVALANCHE":"COL","AVS":"COL","COLORADO AVALANCHE":"COL",

    # --- DAL (Stars) ---
    "DAL":"DAL","DALLAS":"DAL","STARS":"DAL","DALLAS STARS":"DAL",

    # --- DET (Red Wings) ---
    "DET":"DET","DETROIT":"DET","RED WINGS":"DET","REDWINGS":"DET","WINGS":"DET","DETROIT RED WINGS":"DET",

    # --- EDM (Oilers) ---
    "EDM":"EDM","EDMONTON":"EDM","OILERS":"EDM","EDMONTON OILERS":"EDM",

    # --- FLA (Panthers) ---
    "FLA":"FLA","FLORIDA":"FLA","PANTHERS":"FLA","FLORIDA PANTHERS":"FLA",

    # --- LAK (Kings) ---
    "LAK":"LAK","LA":"LAK","L.A.":"LAK","LOS ANGELES":"LAK",
    "KINGS":"LAK","LA KINGS":"LAK","L.A. KINGS":"LAK","LOS ANGELES KINGS":"LAK",

    # --- MIN (Wild) ---
    "MIN":"MIN","MINNESOTA":"MIN","WILD":"MIN","MINNESOTA WILD":"MIN",

    # --- MTL (Canadiens) ---
    "MTL":"MTL","MON":"MTL","MONTREAL":"MTL","CANADIENS":"MTL","HABS":"MTL","MONTREAL CANADIENS":"MTL", "CANADIANS":"MTL", "CANADIAN":"MTL", 

    # --- NJD (Devils) ---
    "NJD":"NJD","NJ":"NJD","N.J.":"NJD","NEW JERSEY":"NJD",
    "DEVILS":"NJD","NEW JERSEY DEVILS":"NJD",

    # --- NSH (Predators) ---
    "NSH":"NSH","NAS":"NSH","NASHVILLE":"NSH","PREDATORS":"NSH","PREDS":"NSH","NASHVILLE PREDATORS":"NSH",

    # --- NYI (Islanders) ---
    "NYI":"NYI","NY ISLANDERS":"NYI","N.Y. ISLANDERS":"NYI","NEW YORK ISLANDERS":"NYI",
    "ISLANDERS":"NYI","ISLES":"NYI",

    # --- NYR (Rangers) ---
    "NYR":"NYR","NY RANGERS":"NYR","N.Y. RANGERS":"NYR","NEW YORK RANGERS":"NYR",
    "RANGERS":"NYR",

    # --- OTT (Senators) ---
    "OTT":"OTT","OTTAWA":"OTT","SENATORS":"OTT","SENS":"OTT","OTTAWA SENATORS":"OTT",

    # --- PHI (Flyers) ---
    "PHI":"PHI","PHILADELPHIA":"PHI","FLYERS":"PHI","PHILADELPHIA FLYERS":"PHI",

    # --- PIT (Penguins) ---
    "PIT":"PIT","PITTSBURGH":"PIT","PENGUINS":"PIT","PENS":"PIT","PITTSBURGH PENGUINS":"PIT",

    # --- SEA (Kraken) ---
    "SEA":"SEA","SEATTLE":"SEA","KRAKEN":"SEA","SEATTLE KRAKEN":"SEA",

    # --- SJS (Sharks) ---
    "SJS":"SJS","SJ":"SJS","S.J.":"SJS","SAN JOSE":"SJS","SANJOSE":"SJS",
    "SHARKS":"SJS","SAN JOSE SHARKS":"SJS","SANJOSE SHARKS":"SJS",

    # --- STL (Blues) ---
    "STL":"STL","ST. LOUIS":"STL","ST LOUIS":"STL","SAINT LOUIS":"STL",
    "BLUES":"STL","ST LOUIS BLUES":"STL","ST. LOUIS BLUES":"STL",

    # --- TBL (Lightning) ---
    "TBL":"TBL","TB":"TBL","T.B.":"TBL","TAMPA BAY":"TBL","TAMPABAY":"TBL",
    "LIGHTNING":"TBL","BOLTS":"TBL","TAMPA BAY LIGHTNING":"TBL","TAMPABAY LIGHTNING":"TBL",

    # --- TOR (Maple Leafs) ---
    "TOR":"TOR","TORONTO":"TOR","MAPLE LEAFS":"TOR","MAPLELEAFS":"TOR","LEAFS":"TOR",
    "TORONTO MAPLE LEAFS":"TOR",

    # --- VAN (Canucks) ---
    "VAN":"VAN","VANCOUVER":"VAN","CANUCKS":"VAN","VANCOUVER CANUCKS":"VAN",

    # --- VGK (Golden Knights) ---
    "VGK":"VGK","VEGAS":"VGK","LAS VEGAS":"VGK","GOLDEN KNIGHTS":"VGK","GOLDENKNIGHTS":"VGK",
    "VEGAS GOLDEN KNIGHTS":"VGK","LAS VEGAS GOLDEN KNIGHTS":"VGK","KNIGHTS":"VGK",

    # --- WPG (Jets) ---
    "WPG":"WPG","WINNIPEG":"WPG","JETS":"WPG","WINNIPEG JETS":"WPG",

    # --- WSH (Capitals) ---
    "WSH":"WSH","WAS":"WSH","WASHINGTON":"WSH","CAPITALS":"WSH","CAPS":"WSH","WASHINGTON CAPITALS":"WSH",

    # --- UTA (Utah Mammoth) ---
    # 실전 표기 변형: Utah / Utah HC / Utah Hockey Club / Utah Mammoth / Mammoth 등
    "UTA":"UTA",
    "UTAH":"UTA",
    "UTAH HC":"UTA","UTAH H.C.":"UTA",
    "UTAH HOCKEY CLUB":"UTA","UTAHHC":"UTA","UTAHHOCKEYCLUB":"UTA",
    "UTAH MAMMOTH":"UTA","UTAH MAMMOTHS":"UTA",
    "MAMMOTH":"UTA","MAMMOTHS":"UTA",
}

FULLNAME = {
    "ANAHEIMDUCKS":"ANA",
    "BOSTONBRUINS":"BOS",
    "BUFFALOSABRES":"BUF",
    "CAROLINAHURRICANES":"CAR",
    "COLUMBUSBLUEJACKETS":"CBJ",
    "CALGARYFLAMES":"CGY",
    "CHICAGOBLACKHAWKS":"CHI",
    "COLORADOAVALANCHE":"COL",
    "DALLASSTARS":"DAL",
    "DETROITREDWINGS":"DET",
    "EDMONTONOILERS":"EDM",
    "FLORIDAPANTHERS":"FLA",
    "LOSANGELESKINGS":"LAK",
    "MINNESOTAWILD":"MIN",
    "MONTREALCANADIENS":"MTL",
    "MONTREALCANADIANS":"MTL",
    "CANADIANS":"MTL",
    "CANADIAN":"MTL",
    "NEWJERSEYDEVILS":"NJD",
    "NASHVILLEPREDATORS":"NSH",
    "NEWYORKISLANDERS":"NYI",
    "NEWYORKRANGERS":"NYR",
    "OTTAWASENATORS":"OTT",
    "PHILADELPHIAFLYERS":"PHI",
    "PITTSBURGHPENGUINS":"PIT",
    "SEATTLEKRAKEN":"SEA",
    "SANJOSESHARKS":"SJS",
    "STLOUISBLUES":"STL",
    "TAMPABAYLIGHTNING":"TBL",
    "TORONTOMAPLELEAFS":"TOR",
    "VANCOUVERCANUCKS":"VAN",
    "VEGASGOLDENKNIGHTS":"VGK",
    "WINNIPEGJETS":"WPG",
    "WASHINGTONCAPITALS":"WSH",

    # Utah Mammoth
    "UTAHMAMMOTH":"UTA",
    "UTAHMAMMOTHS":"UTA",
    "UTAHHOCKEYCLUB":"UTA",
}

def _norm(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_compact(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def to_abbr(x) -> str | None:
    """
    Robust team -> canonical NHL ABBR (32 teams).
    Accepts: already-abbr, "NY Rangers", "New York Rangers", "Rangers",
             "Tampa Bay Lightning", "TB", "Bolts", "Vegas", "Golden Knights",
             "Utah Mammoth", etc.
    """
    if x is None:
        return None

    s = str(x)
    if not s.strip():
        return None

    # 1) exact-like
    s1 = _norm(s)
    if s1 in ABBR:
        return s1
    if s1 in ALIASES:
        return ALIASES[s1]

    # 2) compact normalize
    sc = _norm_compact(s)
    if sc in ABBR:
        return sc
    if sc in ALIASES:
        return ALIASES[sc]
    if sc in FULLNAME:
        return FULLNAME[sc]

    # 3) last chance: pattern heuristics
    if "NEWYORK" in sc and "RANG" in sc:
        return "NYR"
    if "NEWYORK" in sc and "ISLAND" in sc:
        return "NYI"
    if "TAMPABAY" in sc or sc == "TB":
        return "TBL"
    if "SANJOSE" in sc:
        return "SJS"
    if "STLOUIS" in sc or "SAINTLOUIS" in sc:
        return "STL"
    if "LOSANGELES" in sc and "KINGS" in sc:
        return "LAK"
    if "VEGAS" in sc or "GOLDENKNIGHTS" in sc:
        return "VGK"
    if "UTAH" in sc or "MAMMOTH" in sc:
        return "UTA"

    return None