"""Word list for experiment 4: which language does the model "think in" when it writes Kurdish?

Each row is one everyday concept, written in:
  en  English            fr  French (used as the input language)   zh  Chinese
  tr  Turkish            fa  Persian                               ar  Arabic
  kmr     Kurmanji / Badini in Latin letters
  kmr_ar  Badini (Kurmanji) in Arabic letters, as written in Duhok
  ckb     Sorani (Central Kurdish), Arabic letters

The Kurdish words were drafted by an AI (Claude). The Badini words (Arabic letters) were then checked and
corrected by a native Badini speaker (Nashat), and the Latin-letter Kurmanji column was changed to match.
The Sorani column has not been checked by a native speaker.
Several accepted forms can be given with "/": the first one is used in prompts, any of them counts as correct.
"""

COLUMNS = ["en", "fr", "zh", "tr", "fa", "ar", "kmr", "kmr_ar", "ckb"]

WORDS = [
    # en        fr           zh      tr           fa         ar        kmr        kmr_ar      ckb
    ("water",   "eau",       "水",   "su",        "آب",      "ماء",    "av",      "ئاڤ",      "ئاو"),
    ("bread",   "pain",      "面包", "ekmek",     "نان",     "خبز",    "nan",     "نان",      "نان"),
    ("house",   "maison",    "房子", "ev",        "خانه",    "بيت",    "mal",     "مال",      "ماڵ"),
    ("dog",     "chien",     "狗",   "köpek",     "سگ",      "كلب",    "kûçik",   "کووچک",    "سەگ"),
    ("cat", "chat", "猫", "kedi", "گربه", "قطة", "pişîk", "پشیک", "پشیلە"),
    ("sun",     "soleil",    "太阳", "güneş",     "خورشید",  "شمس",    "roj",     "ڕۆژ",      "خۆر"),
    ("moon",    "lune",      "月亮", "ay",        "ماه",     "قمر",    "heyv",    "هەیڤ",     "مانگ"),
    ("tree",    "arbre",     "树",   "ağaç",      "درخت",    "شجرة",   "dar",     "دار",      "دار"),
    ("fire",    "feu",       "火",   "ateş",      "آتش",     "نار",    "agir",    "ئاگر",     "ئاگر"),
    ("eye",     "œil",       "眼睛", "göz",       "چشم",     "عين",    "çav",     "چاڤ",      "چاو"),
    ("hand",    "main",      "手",   "el",        "دست",     "يد",     "dest",    "دەست",     "دەست"),
    ("head", "tête", "头", "baş", "سر", "رأس", "ser", "سەر", "سەر"),
    ("heart",   "cœur",      "心",   "kalp",      "قلب",     "قلب",    "dil",     "دل",       "دڵ"),
    ("blood",   "sang",      "血",   "kan",       "خون",     "دم",     "xwîn",    "خوین",     "خوێن"),
    ("milk",    "lait",      "牛奶", "süt",       "شیر",     "حليب",   "şîr",     "شیر",      "شیر"),
    ("egg",     "œuf",       "鸡蛋", "yumurta",   "تخم",     "بيض",    "hêk",     "هێک",      "هێلکە"),
    ("apple",   "pomme",     "苹果", "elma",      "سیب",     "تفاح",   "sêv",     "سێڤ",      "سێو"),
    ("flower",  "fleur",     "花",   "çiçek",     "گل",      "زهرة",   "gul",     "گول",      "گوڵ"),
    ("mountain", "montagne", "山",   "dağ",       "کوه",     "جبل",    "çiya",    "چیا",      "چیا"),
    ("river",   "rivière",   "河",   "nehir",     "رودخانه", "نهر",    "çem",     "چەم",      "ڕووبار"),
    ("stone",   "pierre",    "石头", "taş",       "سنگ",     "حجر",    "kevir",   "کەڤر",     "بەرد"),
    ("snow",    "neige",     "雪",   "kar",       "برف",     "ثلج",    "berf",    "بەفر",     "بەفر"),
    ("rain",    "pluie",     "雨",   "yağmur",    "باران",   "مطر",    "baran",   "باران",    "باران"),
    ("sea",     "mer",       "海",   "deniz",     "دریا",    "بحر",    "derya",   "دەریا",    "دەریا"),
    ("star",    "étoile",    "星星", "yıldız",    "ستاره",   "نجمة",   "stêr",    "ستێر",     "ئەستێرە"),
    ("sky", "ciel", "天空", "gökyüzü", "آسمان", "سماء", "esman", "ئەسمان", "ئاسمان"),
    ("horse",   "cheval",    "马",   "at",        "اسب",     "حصان",   "hesp",    "هەسپ",     "ئەسپ"),
    ("cow", "vache", "牛", "inek", "گاو", "بقرة", "çêl", "چێل", "مانگا"),
    ("bird", "oiseau", "鸟", "kuş", "پرنده", "طائر", "çûçik/çwîçik/çîçik", "چووچک/چویچک/چیچک", "باڵندە"),
    ("fish",    "poisson",   "鱼",   "balık",     "ماهی",    "سمك",    "masî",    "ماسی",     "ماسی"),
    ("door",    "porte",     "门",   "kapı",      "در",      "باب",    "derî",    "دەری",     "دەرگا"),
    ("book",    "livre",     "书",   "kitap",     "کتاب",    "كتاب",   "pirtûk",  "پرتووک",   "کتێب"),
    ("school", "école", "学校", "okul", "مدرسه", "مدرسة", "qutabxane", "قوتابخانە", "قوتابخانە"),
    ("tea",     "thé",       "茶",   "çay",       "چای",     "شاي",    "çay",     "چای",      "چا"),
    ("salt",    "sel",       "盐",   "tuz",       "نمک",     "ملح",    "xwê",     "خوێ",      "خوێ"),
    ("meat",    "viande",    "肉",   "et",        "گوشت",    "لحم",    "goşt",    "گۆشت",     "گۆشت"),
    ("child",   "enfant",    "孩子", "çocuk",     "کودک",    "طفل",    "zarok",   "زارۆک",    "منداڵ"),
    ("woman",   "femme",     "女人", "kadın",     "زن",      "امرأة",  "jin",     "ژن",       "ژن"),
    ("man",     "homme",     "男人", "adam",      "مرد",     "رجل",    "mêr",     "مێر",      "پیاو"),
    ("mother",  "mère",      "母亲", "anne",      "مادر",    "أم",     "dayik",   "دایک",     "دایک"),
    ("father", "père", "父亲", "baba", "پدر", "أب", "bab", "باب", "باوک"),
    ("brother", "frère",     "兄弟", "kardeş",    "برادر",   "أخ",     "bira",    "برا",      "برا"),
    ("sister",  "sœur",      "姐妹", "abla",      "خواهر",   "أخت",    "xwişk",   "خویشک",    "خوشک"),
    ("friend",  "ami",       "朋友", "arkadaş",   "دوست",    "صديق",   "heval",   "هەڤاڵ",    "هاوڕێ"),
    ("name",    "nom",       "名字", "isim",      "نام",     "اسم",    "nav",     "ناڤ",      "ناو"),
    ("night",   "nuit",      "夜",   "gece",      "شب",      "ليل",    "şev",     "شەڤ",      "شەو"),
    ("year",    "année",     "年",   "yıl",       "سال",     "سنة",    "sal",     "ساڵ",      "ساڵ"),
    ("village", "village",   "村庄", "köy",       "روستا",   "قرية",   "gund",    "گوند",     "گوند"),
    ("city",    "ville",     "城市", "şehir",     "شهر",     "مدينة",  "bajar",   "باژێر",    "شار"),
    ("road",    "route",     "路",   "yol",       "راه",     "طريق",   "rê",      "ڕێ",       "ڕێگا"),
    ("song",    "chanson",   "歌",   "şarkı",     "آهنگ",    "أغنية",  "stran",   "ستران",    "گۆرانی"),
    ("dream",   "rêve",      "梦",   "rüya",      "خواب",    "حلم",    "xewn",    "خەون",     "خەون"),
    ("love",    "amour",     "爱",   "aşk",       "عشق",     "حب",     "evîn",    "ئەڤین",    "خۆشەویستی"),
    ("death",   "mort",      "死亡", "ölüm",      "مرگ",     "موت",    "mirin",   "مرن",      "مردن"),
    ("language", "langue",   "语言", "dil",       "زبان",    "لغة",    "ziman",   "زمان",     "زمان"),
    ("king", "roi", "国王", "kral", "شاه", "ملك", "paşa", "پاشا", "پاشا"),
    ("gold",    "or",        "金",   "altın",     "طلا",     "ذهب",    "zêr",     "زێڕ",      "زێڕ"),
    ("wolf", "loup", "狼", "kurt", "گرگ", "ذئب", "gurg", "گورگ", "گورگ"),
    ("knife", "couteau", "刀", "bıçak", "چاقو", "سكين", "kêrik", "کێرک", "چەقۆ"),
    ("grape",   "raisin",    "葡萄", "üzüm",      "انگور",   "عنب",    "tirî",    "تری",      "ترێ"),
    ("honey", "miel", "蜂蜜", "bal", "عسل", "عسل", "hingivîn", "هنگڤین", "هەنگوین"),
    ("key",     "clé",       "钥匙", "anahtar",   "کلید",    "مفتاح",  "kilîl",   "کلیل",     "کلیل"),
    ("shoe",    "chaussure", "鞋",   "ayakkabı",  "کفش",     "حذاء",   "pêlav",   "پێلاڤ",    "پێڵاو"),
    ("bridge",  "pont",      "桥",   "köprü",     "پل",      "جسر",    "pir",     "پر",       "پرد"),
    ("garden",  "jardin",    "花园", "bahçe",     "باغ",     "حديقة",  "baxçe",   "باخچە",    "باخچە"),
    ("wheat",   "blé",       "小麦", "buğday",    "گندم",    "قمح",    "genim",   "گەنم",     "گەنم"),
    ("thread", "fil", "线", "iplik", "نخ", "خيط", "benik", "بەنک", "دەزوو"),
    ("tooth", "dent", "牙齿", "diş", "دندان", "سن", "didan", "ددان", "ددان"),
    ("ear",     "oreille",   "耳朵", "kulak",     "گوش",     "أذن",    "guh",     "گوھ",      "گوێ"),
    ("cloud", "nuage", "云", "bulut", "ابر", "سحابة", "ewr", "عەور", "هەور"),
]

# Words kept apart and shown as examples inside the prompt (never tested).
N_EXAMPLES = 5

LANG_NAMES = {  # how each language names itself in the prompt
    "en": "English", "fr": "Français", "zh": "中文", "tr": "Türkçe", "fa": "فارسی", "ar": "العربية",
    "kmr": "Kurmancî", "kmr_ar": "کورمانجی", "ckb": "کوردی",
}
LANG_LABELS = {  # for charts and the report
    "en": "English", "fr": "French (input)", "zh": "Chinese", "tr": "Turkish", "fa": "Persian", "ar": "Arabic",
    "kmr": "Kurmanji (Latin letters)", "kmr_ar": "Badini (Arabic letters)", "ckb": "Sorani",
}
