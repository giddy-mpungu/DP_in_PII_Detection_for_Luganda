"""Synthetic, public Luganda PII sentences for the warm start (Experiment E).

Nothing here is derived from lg-ner sentences: names, places, organisations, groups and identifier formats are
general public knowledge (common Ugandan names, districts, public bodies, political parties, the Ugandan mobile
numbering plan, the 14-character national ID layout), and the sentence frames were written for this purpose.
Two FORMAT conventions follow the corpus so that the warm-start model matches its input and labels:
  * text is lower-cased, and long numbers / ID strings are split into word pieces ("0 ##7 ##7 ...");
  * an entity whose final piece is a single character leaves that piece untagged (dissertation Section 4.8).
Because no private sentence is used, training on this data costs no privacy budget.

usage: python synthetic_data.py [n] [seed]     -> prints a few examples
"""
import random

FIRST = ["musoke", "nakato", "babirye", "kato", "waswa", "nankya", "ssemakula", "namukasa", "mukasa", "nalwoga",
         "kiggundu", "nansubuga", "lubega", "kawooya", "ssenyonga", "nabukenya", "kizito", "namatovu", "ssali",
         "nakimuli", "mugerwa", "nalubega", "kasozi", "nakawuki", "ssebuliba", "nabatanzi", "walusimbi", "nanyonga",
         "peter", "sarah", "john", "grace", "joseph", "esther", "david", "agnes", "moses", "ruth", "robert", "brenda",
         "isaac", "florence", "paul", "juliet", "henry", "rose", "ibrahim", "aisha", "hassan", "zainab", "yusuf",
         "fatuma", "charles", "harriet", "emmanuel", "betty", "ronald", "prossy", "fred", "annet"]
LAST = ["mukiibi", "kasule", "ssempala", "lwanga", "kigozi", "nsubuga", "ssekandi", "kayongo", "kaggwa", "bbosa",
        "ssewankambo", "kalule", "mutebi", "kibirige", "lukwago", "sserwadda", "kakooza", "nakabugo", "namubiru",
        "kabuye", "ssentongo", "mayanja", "ssebunya", "kamya", "muwanga", "sekitoleko", "zziwa", "musisi"]
TITLE = ["", "", "", "", "omukyala ", "omwami ", "dr ", "omubaka ", "omusumba ", "sheikh ", "maama ", "taata "]
PLACE = ["kampala", "entebbe", "masaka", "mukono", "jinja", "wakiso", "mityana", "mubende", "luweero", "mpigi",
         "kayunga", "nakasongola", "kalangala", "rakai", "lyantonde", "mbarara", "gulu", "lira", "mbale", "hoima",
         "fort portal", "kabale", "arua", "soroti", "tororo", "ntinda", "nakawa", "kawempe", "rubaga", "makindye",
         "kireka", "bweyogerere", "kajjansi", "nansana", "kyengera", "busega", "natete", "wandegeya", "kisenyi",
         "katwe", "nakulabye", "kenya", "tanzania", "rwanda", "south sudan", "congo", "buganda", "busoga", "ankole"]
ORG = ["kcca", "ura", "nssf", "updf", "makerere university", "mulago hospital", "bank of uganda", "mtn uganda",
       "airtel uganda", "stanbic bank", "uganda police", "nira", "kyambogo university", "umeme", "nwsc",
       "uganda revenue authority", "centenary bank", "radio simba", "bukedde", "ministry of health",
       "electoral commission", "uganda cranes", "red cross", "unicef", "who"]
NORP = ["nrm", "fdc", "nup", "dp", "upc", "ant", "abaganda", "abanyankole", "abasoga", "abagisu", "abacholi",
        "abakatoliki", "abasiraamu", "abalokole", "abaprotestanti", "abanyarwanda", "abakenya", "bayudaaya",
        "abanyamulenga", "abalangi", "abateeso"]
WEEKDAY = ["bbalaza", "lwakubiri", "lwakusatu", "lwakuna", "lwakutaano", "lwamukaaga", "ssande",
           "monday", "tuesday", "friday", "sunday"]
MONTH = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
         "november", "december", "janwali", "febwali", "maaki", "apuli", "maayi", "juuni", "julaayi", "agusito",
         "sebutemba", "okitobba", "novemba", "desemba"]
ORDINAL = ["okubiri", "okusatu", "okuna", "okutaano", "omukaaga", "omusanvu", "omunaana", "omwenda"]
EMAIL_HOST = ["gmail", "yahoo", "outlook"]

# frames: {X} is a slot; plain words are O. Frames with no PII teach the model what NOT to flag.
FRAMES = [
    "{PER} yatuuse e {LOC} ku {DATE} okusisinkana abakulembeze ba {ORG} .",
    "ennamba y ' essimu ya {PER} ye {PHONE} .",
    "kuba ku {PHONE} okufuna obuyambi obw ' amangu .",
    "{PER} alina ndagamuntu ennamba {NIN} .",
    "ndagamuntu ya {PER} eriko ennamba {NIN} , era abeera e {LOC} .",
    "emmotoka ennamba {PLATE} ekwatiddwa poliisi e {LOC} .",
    "pikipiki ennamba {PLATE} yabbiddwa ku {DATE} e {LOC} .",
    "ababaka ba {NORP} bakubye olukung ' aana e {LOC} ku {DATE} .",
    "{PER} owa {NORP} agambye nti {ORG} yaakuyamba abantu b ' e {LOC} .",
    "mu {DATE} , {ORG} yafulumizza alipoota empya ku byenfuna .",
    "{PER} yazaalibwa nga {DATE} mu disitulikiti y ' e {LOC} .",
    "{PER} yakwatiddwa ku {DATE} ng ' alina ndagamuntu {NIN} .",
    "abantu ba {NORP} abasoba mu lukumi beetabye mu kusaba e {LOC} .",
    "olukiiko lwa {ORG} lwakutuula ku {DATE} e {LOC} .",
    "{PER} ne {PER} basisinkanye mu woofiisi za {ORG} e {LOC} .",
    "email ya {PER} ye {EMAIL} .",
    "weereza ssente ku {PHONE} oba ku {PHONE} .",
    "{PER} akola ne {ORG} okuva mu {DATE} .",
    "abavubuka ba {NORP} e {LOC} basabye gavumenti ebayambe .",
    "{PER} yasabye abantu okumukubira ku {PHONE} oba okuwandiikira {EMAIL} .",
    "akabenje kaagudde ku luguudo lw ' e {LOC} ku {DATE} , emmotoka {PLATE} n ' etomera pikipiki .",
    "{ORG} erangiridde nti ebigezo bijja kutandika ku {DATE} .",
    "omusawo {PER} owa {ORG} agambye nti abalwadde beeyongedde e {LOC} .",
    "ssente {AMOUNT} ze zaasasuddwa abakozi ba {ORG} .",
    "abantu {COUNT} be baafiiridde mu mataba e {LOC} .",
    "omupiira gwa {ORG} gwakuzannyibwa ku {DATE} .",
    "ebbeeyi y ' amafuta yalinnye okutuuka ku ssiringi {AMOUNT} .",
    "abayizi {COUNT} be baatuula ebigezo omwaka guno .",
    "gavumenti egamba nti amakubo {COUNT} gaakuddabirizibwa .",
    "{PER} ow ' emyaka {COUNT} yafunye omulimu mu {ORG} .",
]


def pieces(s, rng, digit_split=True):
    """Corpus-style word pieces: digits one per piece, letter runs in 1-3 character pieces."""
    out, i = [], 0
    while i < len(s):
        if s[i].isdigit() and digit_split:
            out.append(s[i]); i += 1
        else:
            j = i
            while j < len(s) and not s[j].isdigit():
                j += 1
            run = s[i:j]
            while run:
                k = min(len(run), rng.randint(1, 3))
                out.append(run[:k]); run = run[k:]
            i = j
    return [p if n == 0 else "##" + p for n, p in enumerate(out)]


def slot(kind, rng):
    """-> list of (word pieces of one entity mention) as a token list, entity type or None"""
    if kind == "PER":
        name = rng.choice(FIRST) + (" " + rng.choice(LAST) if rng.random() < 0.7 else "")
        title = rng.choice(TITLE).split()
        return title, name.split(), "PERSON"
    if kind == "LOC":
        return [], rng.choice(PLACE).split(), "LOCATION"
    if kind == "ORG":
        return [], rng.choice(ORG).split(), "ORG"
    if kind == "NORP":
        return [], rng.choice(NORP).split(), "NORP"
    if kind == "DATE":
        year = str(rng.randint(1960, 2026))
        yr = [year] if rng.random() < 0.5 else [year[:3], "##" + year[3]]
        day = str(rng.randint(1, 31))
        dy = [day] if len(day) == 1 or rng.random() < 0.5 else [day[0], "##" + day[1]]
        form = rng.randrange(7)
        if form == 0:
            return [], [rng.choice(WEEKDAY)], "DATE"
        if form == 1:
            return [], dy + [rng.choice(MONTH)] + yr, "DATE"
        if form == 2:
            return [], yr, "DATE"
        if form == 3:
            return [], ["mwezi", "gw", "'", rng.choice(ORDINAL)] + yr, "DATE"
        if form == 4:
            return [], [rng.choice(WEEKDAY), "lwa", "wiiki", rng.choice(["eno", "ewedde", "ejja"])], "DATE"
        if form == 5:
            return [], dy + ["/", f"{rng.randint(1, 12):02d}", "/"] + yr, "DATE"
        return [], [rng.choice(MONTH)] + yr, "DATE"
    if kind == "PHONE":
        num = rng.choice(["07", "2567"]) + "".join(str(rng.randrange(10)) for _ in range(8))
        return [], pieces(num, rng), "USERID"
    if kind == "NIN":
        letters = lambda n: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        nin = letters(2) + "".join(str(rng.randrange(10)) for _ in range(8)) + letters(4)
        return [], pieces(nin, rng), "USERID"
    if kind == "PLATE":
        plate = "u" + "".join(rng.choice("abcdefghjklmnprstuvwxyz") for _ in range(2)) + \
                "".join(str(rng.randrange(10)) for _ in range(3)) + rng.choice("abcdefghjklmnprstuvwxyz")
        return [], pieces(plate, rng), "USERID"
    if kind == "EMAIL":
        user = rng.choice(FIRST) + rng.choice(["", ".", "_"]) + rng.choice(LAST) + str(rng.randint(1, 99))
        return [], user.replace(".", " . ").replace("_", " _ ").split() + ["@", rng.choice(EMAIL_HOST), ".", "com"], "USERID"
    if kind == "AMOUNT":
        return [], pieces(str(rng.choice([5, 10, 20, 50, 100, 250, 500]) * 1000), rng), None
    if kind == "COUNT":
        return [], [str(rng.randint(2, 99))], None
    raise KeyError(kind)


def bilou(n, typ):
    return [f"U-{typ}"] if n == 1 else [f"B-{typ}"] + [f"I-{typ}"] * (n - 2) + [f"L-{typ}"]


def sentence(rng):
    toks, tags = [], []
    for part in rng.choice(FRAMES).split():
        if part.startswith("{") and part.endswith("}"):
            pre, ent, typ = slot(part[1:-1], rng)
            toks += pre; tags += ["O"] * len(pre)
            if typ is None:
                toks += ent; tags += ["O"] * len(ent)
                continue
            n = len(ent)
            if n > 1 and len(ent[-1].lstrip("#")) == 1:   # corpus convention: final 1-character piece untagged
                tags += bilou(n - 1, typ) + ["O"]
            else:
                tags += bilou(n, typ)
            toks += ent
        else:
            toks.append(part); tags.append("O")
    return toks, tags


def generate(n=3000, seed=0):
    rng = random.Random(seed)
    return [sentence(rng) for _ in range(n)]


if __name__ == "__main__":
    import sys
    n, seed = (int(sys.argv[1]) if len(sys.argv) > 1 else 8), (int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    for toks, tags in generate(n, seed):
        print(" ".join(t if g == "O" else f"{t}/{g}" for t, g in zip(toks, tags)))
